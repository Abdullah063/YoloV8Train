import yaml

data_yaml = {
    'path': '/Users/altun/Desktop/lostProject/DataSet/dataSet2/obj',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 15,
    'names': ['elma', 'kivi', 'mango', 'guava', 'muz', 'portakal', 'armut', 'şeftali',
              'pitaya', 'erik', 'domates', 'nar', 'karambola', 'kavun', 'hurma']
}

with open('dataset.yaml', 'w') as f:
    yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)