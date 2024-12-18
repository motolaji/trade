from pathlib import Path
from tqdm import tqdm
import shutil


image_dir = Path('C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/images')
dataset_path = Path('C:/Users/jajim/Desktop/dissertation/trade/L3D-dataset/dataset/images')
image_paths = list(dataset_path.glob('*.[jp][pn][g]'))

# save_dir = Path('L3D-dataset')
# save_dir.mkdir(exist_ok=True)

dst = Path('new_images_dataset')
dst.mkdir(exist_ok=True)

files = sorted(
    [f for f in image_dir.glob('*') if f.is_file()],
    key=lambda x: x.stat().st_size,
)
    
files = list(image_dir.rglob('*'))
for file in tqdm(files[:100000], desc='Copying images'):
   file.rename(dst / file.name)

# print(len(image_paths))


# do not run !!!