import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import tqdm


def create_directory_structure(base_path):
    """Gerekli klasör yapısını oluşturur"""
    directories = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]

    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        Path(full_path).mkdir(parents=True, exist_ok=True)
        print(f"Oluşturulan klasör: {full_path}")


def organize_dataset(source_img_dir, source_label_dir, output_base_dir,
                     train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Veri setini train, validation ve test olarak böler ve yeni yapıya kopyalar

    Args:
        source_img_dir: Kaynak görüntü klasörü
        source_label_dir: Kaynak etiket klasörü
        output_base_dir: Çıktı klasörü
        train_ratio: Eğitim seti oranı
        val_ratio: Doğrulama seti oranı
        test_ratio: Test seti oranı
    """

    # Klasör yapısını oluştur
    create_directory_structure(output_base_dir)

    # Tüm görüntü dosyalarını listele
    image_files = [f for f in os.listdir(source_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Görüntüleri train, validation ve test olarak böl
    train_files, remaining = train_test_split(image_files, train_size=train_ratio, random_state=42)

    # Kalan dosyaları validation ve test olarak böl
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(remaining, train_size=relative_val_ratio, random_state=42)

    # Dosyaları kopyala
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    print("\nDosya dağılımı:")
    print(f"Eğitim seti: {len(train_files)} dosya")
    print(f"Doğrulama seti: {len(val_files)} dosya")
    print(f"Test seti: {len(test_files)} dosya")

    # Her split için dosyaları kopyala
    for split_name, files in splits.items():
        print(f"\n{split_name} seti için dosyalar kopyalanıyor...")

        for img_file in tqdm.tqdm(files):
            # Görüntü dosyası için işlemler
            src_img = os.path.join(source_img_dir, img_file)
            dst_img = os.path.join(output_base_dir, 'images', split_name, img_file)
            shutil.copy2(src_img, dst_img)

            # Etiket dosyası için işlemler
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(source_label_dir, label_file)
            dst_label = os.path.join(output_base_dir, 'labels', split_name, label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Uyarı: {label_file} etiket dosyası bulunamadı!")


def create_dataset_yaml(output_base_dir, num_classes=1):
    """YOLO için dataset.yaml dosyası oluşturur"""
    yaml_content = f"""
path: {output_base_dir}  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images (optional)

names:
  0: fruit  # class names
""".strip()

    yaml_path = os.path.join(output_base_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\ndataset.yaml dosyası oluşturuldu: {yaml_path}")


def main():
    # Kaynak ve hedef dizinleri
    source_img_dir = '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj/img'  # Kaynak görüntü klasörü
    source_label_dir = '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj/label'  # Kaynak etiket klasörü
    output_base_dir = '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj'  # Çıktı klasörü

    # Veri setini organize et
    organize_dataset(source_img_dir, source_label_dir, output_base_dir)

    # YOLO config dosyasını oluştur
    create_dataset_yaml(output_base_dir)


if __name__ == "__main__":
    main()