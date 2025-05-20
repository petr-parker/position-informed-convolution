import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pycocotools.mask import decode as coco2mask
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A


class SAMDS(Dataset):
    '''
    Датасет SAMv1
    '''
    @staticmethod
    def img_name2annotations(img_name):
        '''
        Читает разметку для текущего изображения:
        '''
        with open(img_name + '.json', 'r') as file:
            return json.load(file)['annotations']

    @classmethod
    def indexation(cls, ds_path, num_images_limit=None):
        '''
        Индексация данных.
        '''

        # Список файлов в папке с датасетом:
        file_names = os.listdir(ds_path)

        # Индексируем изображения и их маски:
        img_names = []
        mask_inds = []
        for file in tqdm(file_names, desc='Индексация датасета'):

            # Уточняем путь до файла:
            file = os.path.join(ds_path, file)

            # Расщепляем путь до файла на имя и расширение:
            img_name, ext = os.path.splitext(file)

            # Если файл не является изображением, или у него нет
            # соответствующей разметки, то пропускаем:
            if ext != '.jpg':
                continue
            if not os.path.isfile(img_name + '.json'):
                continue

            # Индексируем имя файлов:
            img_names.append(img_name)

            # Индексируем каждую маску для изображения:
            annotations = cls.img_name2annotations(img_name)
            for mask_ind in range(len(annotations)):
                mask_inds.append((img_name, mask_ind))

            # Ограничение на объём индексации, чтобы не тратить слишком много
            # времени во время отладки:
            if num_images_limit and \
                    len(img_names) > num_images_limit:
                break

        return img_names, mask_inds

    def __init__(self,
                 ds_path,
                 imsize=None,
                 rect_noise_rate=0.2,
                 store_img_mask=False,
                 train=True,
                 num_images_limit=None,
                 indexation=None):

        # Фиксируем параметры:
        self.ds_path = ds_path
        self.imsize = imsize
        self.rect_noise_rate = rect_noise_rate
        self.store_img_mask = store_img_mask
        self.train = train
        self.num_images_limit = num_images_limit

        # Инициируем аугментирующие преобразования:
        self._init_transforms()

        # Индексация файлов:
        if indexation is None:
            self.img_names, self.mask_inds = self.indexation(
                self.ds_path, self.num_images_limit
            )

        # Берётся готовая, если задаётся явно из make_train_test:
        else:
            self.img_names, self.mask_inds = indexation

    def as_train_test(self, test_size=0.1, random_state=42):

        # Разделяем список изображений на обучающую и проверочную части:
        train_img_names, test_img_names = train_test_split(
            self.img_names,
            test_size=test_size,
            random_state=random_state
        )

        # Повторяем разделение mask_inds по аналогии с img_names:
        train_mask_inds = []
        test_mask_inds = []
        for img_name, mask_ind in self.mask_inds:
            if img_name in train_img_names:
                train_mask_inds.append((img_name, mask_ind))
            else:
                test_mask_inds.append((img_name, mask_ind))

        # Конструктор текущего класса:
        cls = type(self)

        # Создаём новые экземпляры класса:
        train = cls(ds_path=self.ds_path,
                    imsize=self.imsize,
                    rect_noise_rate=self.rect_noise_rate,
                    store_img_mask=self.store_img_mask,
                    train=self.train,
                    indexation=(train_img_names, train_mask_inds))
        test = cls(ds_path=self.ds_path,
                   imsize=self.imsize,
                   rect_noise_rate=0.,
                   store_img_mask=self.store_img_mask,
                   train=False,
                   indexation=(test_img_names, test_mask_inds))

        return train, test

    def _init_transforms(self):

        # Если нужна аугментация:
        if self.train:

            # Основная аугментация:
            if self.store_img_mask:
                border_kwargs = {}
                transforms = [
                    A.Affine(
                        translate_percent={
                            'x': (-0.1, 0.1),
                            'y': (-0.1, 0.1)
                        },
                        scale=(0.9, 1.1),
                        rotate=(-10, 10),
                        shear=(-10, 10),
                        rotate_method='largest_box',
                        interpolation=cv2.INTER_AREA,
                        p=1.,
                        **border_kwargs
                    ),
                ]
            else:
                transforms = []

            transforms += [
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]

            # Если задано целевое разрешение, дабавляем финальную аугментацию:
            if self.imsize is not None:
                transforms.append(
                    A.RandomSizedBBoxSafeCrop(
                        *self.imsize,
                        interpolation=cv2.INTER_AREA,
                        erosion_rate=0.9
                    )
                )
            # Приходится использовать RandomSizedBBoxSafeCrop, т.к.
            # CropNonEmptyMaskIfExists почему-то пока не работает:
            # github.com/albumentations-team/albumentations/issues/2524

            # Собираем аугментацию:
            transforms = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format='coco')
            )

        # Если сама аугментация не нужна и целевое разрешение не задано:
        elif self.imsize is None:
            transforms = None

        # Если сама аугментация не нужна, но целевое разрешение задано:
        else:
            transforms = A.Resize(*self.imsize, interpolation=cv2.INTER_AREA)

        self.transforms = transforms

    def __len__(self):
        return len(self.img_names if self.train else self.mask_inds)

    @staticmethod
    def mask2noised_rect_mask(mask, rect_noise_rate):
        '''
        По эталонной маске строит маску обрамляющего прямоугольника с небольшими
        ошибками.

        rect_noise_rate - Уровень зашумнелия границ обрамляющего прямоугольника
        (доля от его размеров).
        '''
        # Инициируем маску прямоугольника:
        rect_mask = np.zeros_like(mask)

        # Если эталонная маска пуста, то и итоговую возвращаем пустой:
        if not mask.any():
            return rect_mask

        # Получаем параметры обрамляющего прямоугольника для эталонной маски:
        rect = np.array(cv2.boundingRect(mask))

        # Определяем размеры шумов по горизонтали и вертикали:
        x_nlevel, y_nlevel = (rect[2:] * rect_noise_rate).astype(int)

        # Генерируем шум прямоугольника с равномерным распределением:
        x_noise = np.random.randint(x_nlevel * 2 + 1, size=2) - x_nlevel
        y_noise = np.random.randint(y_nlevel * 2 + 1, size=2) - y_nlevel
        rect_noise = np.array([x_noise[0], y_noise[0], x_noise[1], y_noise[1]])

        # Меняем формат прямоугольника и накладываем шум:
        rect[2:] += rect[:2]  # LTWH -> LTRB
        rect += rect_noise    # Наложение шума

        # Рисуем зашумлённый прямоугольник:
        return cv2.rectangle(rect_mask, rect[:2], rect[2:] - 1, 255, -1)

    def __getitem__(self, idx):

        if self.train:

            # Определяем путь до файлов:
            img_name = self.img_names[idx]

            # Читаем разметку:
            annotations = self.img_name2annotations(img_name)

            # Берём случайную маску:
            annotation = np.random.choice(annotations)

        else:

            # Определяем путь до файлов и номер маски:
            img_name, mask_ind = self.mask_inds[idx]

            # Читаем разметку:
            annotations = self.img_name2annotations(img_name)

            # Берём маску с нужным индексом:
            annotation = annotations[mask_ind]

        # Загружаем изображение в RGB uint8:
        img = cv2.imread(img_name + '.jpg')[..., ::-1]

        # Растеризируем эталонную маску:
        gt_mask = coco2mask(annotation['segmentation'])

        # Добавляем к эталонной маске второй канал, если надо:
        if self.store_img_mask:
            img_mask = np.ones_like(gt_mask) * 255
            mask = np.dstack([gt_mask, img_mask])
            # Вторая маска будет скрывать области изображения, вышедшие за
            # рамки в результате аугментации.
        else:
            mask = gt_mask

        # Применяем аугментацию, если нужно:
        if self.transforms:

            # Формируем список словарь аргументов:
            kwargs = {'image': img, 'mask': mask}

            # Дополняем обрамляющим прямоугольником, если используется
            # RandomSizedBBoxSafeCrop:
            if self.imsize is not None and self.train:
                kwargs['bboxes'] = [annotation['bbox']]

            # Выполняем преобразование:
            transformed = self.transforms(**kwargs)
            img = transformed['image']
            mask = transformed['mask']

        # Если нужна маска трансформации изображения:
        if self.store_img_mask:

            # Извлекаем вторую составляющую из трансформированной маски:
            gt_mask, img_mask = mask[..., 0], mask[..., 1]

        else:
            gt_mask = mask

        # Формируем прямоугольную маску из эталонной, если она есть:
        rect_mask = self.mask2noised_rect_mask(gt_mask, self.rect_noise_rate)

        # Прикрепляем
        if self.store_img_mask:
            img = np.dstack([img, img_mask, rect_mask])
        else:
            img = np.dstack([img, rect_mask])

        # Перевод во Float:
        img = img / 255
        gt_mask.astype(img.dtype)

        return img, gt_mask

    def show_examples(self, idx=None, num=4):

        # Доопределяем номер семпла:
        rand_ind = np.random.randint(len(self)) if idx is None else idx

        # Демонстрируем примеры несколько раз:
        for _ in range(num):

            # Извлекаем семпл:
            img, mask = self[rand_ind]

            # Переносим все маски во второе изображение:
            rgb_img = img[..., :3]
            rgb_mask = np.dstack([img[..., 3:], mask])[..., ::-1]

            # Если маска аугментации не использовалась = дополняем единицами:
            if rgb_mask.shape[2] == 2:
                rgb_mask = np.dstack([rgb_mask, np.ones_like(mask)])

            # Вычетаем из синего два других канала:
            rgb_mask[..., -1] -= rgb_mask[..., :-1].max(-1)
            rgb_mask[rgb_mask < 0] = 0
            # Все отрицательные (в результате вычитания) значения обнуляем.

            plt.figure()
            plt.imshow(np.hstack([rgb_img, rgb_mask]))
            plt.axis(False)


if __name__ == '__main__':
    ds_path = '/outroot/mnt/rsm3/DATA_1/projects_dir/PIC/ds/'
    sam_ds = SAMDS(
        ds_path,
        imsize=(64, 64),
        rect_noise_rate=0.1,
        #train=False,
        #store_img_mask=True,
        num_images_limit=1000,
    )

    train_ds, test_ds = sam_ds.as_train_test(10)

    print('Both:')
    sam_ds.show_examples()
    plt.show()

    print('Train:')
    train_ds.show_examples()
    plt.show()

    print('Test:')
    test_ds.show_examples()
    plt.show()

    print(len(sam_ds), len(train_ds), len(test_ds))