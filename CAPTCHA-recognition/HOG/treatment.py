import platform
import numpy as np
import pandas as pd
import sys,os, math, time
import concurrent.futures, multiprocessing

sys.path.append(os.getcwd())

from utils import get_labels

from PIL import Image

def read_image(image_path):
    image = Image.open(image_path).convert("L")

    image = image.resize((image.size[0] + 12, image.size[1] - 2))
    image = np.array(image).astype(np.uint8)

    return image


def divide_image(image):
    size_part = math.ceil(image.shape[1] / 6)

    parts = []
    for i in range(0, image.shape[1], size_part):
        if i == 0:
            parts.append(image[:,i:i+size_part+16])
        elif i == image.shape[1] - size_part:
            parts.append(image[:,i-16:i+size_part])
        else:
            parts.append(image[:,i-8:i+size_part+8])

    return parts


def apply_filter(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    result = np.zeros((m-km+1, n-kn+1))
    for i in range(m-km+1):
        for j in range(n-kn+1):
            result[i, j] = np.sum(image[i:i+km, j:j+kn] * kernel)
    return result

def cart_to_polar(dx,dy):
  m,n = dx.shape
  mag = np.zeros((m, n))
  angle = np.zeros((m, n))

  for i in range(m):
    for j in range(n):
      mag[i,j] = math.sqrt(dx[i,j]**2+dy[i,j]**2)
      angle[i,j] = math.degrees(math.atan2(dy[i,j],dx[i,j]))
      if angle[i,j] < 0:
        angle[i,j] = 180 + angle[i,j]
      if angle[i,j] == 180:
        angle[i,j] = 0

  return mag,angle

def derivate_image(image):
    # Defina o kernel para derivada em relação às linhas (vetor coluna)
    row_kernel = np.array([[-1], [0], [1]])

    # Defina o kernel para derivada em relação às colunas (vetor linha)
    col_kernel = np.array([[-1, 0, 1]])

    # Aplique a convolução para calcular as derivadas
    m,n = image.shape
    dx = np.zeros((m, n))
    dy = np.zeros((m, n))

    dx[:,1:n-1] = apply_filter(image, col_kernel)
    dy[1:m-1,:] = apply_filter(image, row_kernel)

    return cart_to_polar(dx, dy)

def lala(mag,ang):
  m,n = mag.shape
  hist = np.zeros(9)
  for i in range(m):
    for j in range(n):
      idx_1 = int(ang[i,j] / 20)
      idx_2 = (math.ceil(ang[i,j] / 20)) % 9
      if idx_1 == idx_2:
        hist[idx_1] += mag[i,j]
      else:
        hist[idx_1] += mag[i,j]*(1 - ((ang[i,j] - idx_1*20)/20))
        hist[idx_2] += mag[i,j]*((ang[i,j] - idx_1*20)/20)
  return hist.tolist()

def hist_of_gradient(mag,ang):
  m,n = mag.shape
  lista = []
  for i in range(0,m-4, 4):
    for j in range(0,n-4, 4):
      lista += lala(mag[i:i+8,j:j+8],ang[i:i+8,j:j+8])
  return lista


def execute(filenames):
    for filename in filenames:
      print(f"{len(dataset)} de {len(files)*6}")
      img = read_image(f"CAPTCHA-10k/validacao/{filename}")
      
      parts = divide_image(img)

      label = filename.replace(".jpg", "")
      captcha = labels[label]

      for i, part in enumerate(parts):
          mag, ang = derivate_image(part)

          dataset.append([int(filename.replace(".jpg", ""))*6 + i] + [captcha[i]] + hist_of_gradient(mag, ang))

      if platform.system() == 'Windows':
        os.system('cls')
      else:
        os.system('clear')

if __name__ == "__main__": 
  labels = get_labels()

  dataset = []

  files = sorted(list(os.listdir("CAPTCHA-10k/validacao")))

  workers = multiprocessing.cpu_count()
  print(f"Número de cores: {workers}")
  time.sleep(1)

  args = np.array_split(files, workers)
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
      futures = [executor.submit(execute, arg) for arg in args]

      concurrent.futures.wait(futures)
  
  df = pd.DataFrame(dataset)

  df = df.sort_values(by=0)
  df = df.drop(columns=[0])
  df.columns = df.columns.astype(int) - 1
  df = df.reset_index().drop("index", axis=1)

  df.to_csv("validation.csv")
