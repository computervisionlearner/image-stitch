import argparse
import cv2
import numpy as np
'''
以右图为标准变换
'''

ratio = 0.75
reprojThresh =4.0
def get_the_kps_and_features(image):
  '''
  输入：一张图片
  输出：图片的特征点坐标集和特征点对应的特征向量集
  '''
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  descriptor = cv2.xfeatures2d.SIFT_create()
  raw_kps,features = descriptor.detectAndCompute(gray, None)
  
  return raw_kps, features
  

def match_keypoints(features1, features2):
  '''
  输入：两张图片的特征向量集
  功能：用knn算法进行特征匹配
  输出：找出在features2中与features1中每个特征向量最接近的两个特征向量
  '''
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(features1, features2, k = 2)
  return matches

def find_H(raw_kps1, raw_kps2, raw_matches):
  '''
  输入：两幅图像的特征点对象和特征点匹配集
  功能：通过计算某特征向量的最接近的特征向量是否小于一定比例下的次接近的特征向量，
  来判断是否是最佳匹配点对
  输出：最佳匹配的关键点对、计算单应矩阵H、配对状态
  
  '''
  kps1 = np.float32([kp.pt for kp in raw_kps1])
  kps2 = np.float32([kp.pt for kp in raw_kps2])
  
  good_matches = []
  for m0,m1 in raw_matches:
    if m0.distance <m1.distance * ratio:
      good_matches.append((m0.queryIdx,m0.trainIdx))
      
  if len(good_matches) > 4:
    pts1 = np.float32([kps1[i] for (i,_) in good_matches])
    pts2 = np.float32([kps2[i] for (_,i) in good_matches])
    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC,
                                   reprojThresh)
    return good_matches, H, status
  return None

def stitch(b, a, good_matches, H, status):
  '''
  输入：源图、目的图、最佳匹配点对、单应矩阵H、匹配状态
  输出：拼接后的图
  功能：将源图通过矩阵H变换到目的图所处的坐标系中，然后用目的图覆盖对应的区域，形成拼接后的图像
  '''
  xh = np.linalg.inv(H)

  ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
  ds = ds/ds[-1]

  f1 = np.dot(xh, np.array([0,0,1]))
  f1 = f1/f1[-1]
  xh[0][-1] += abs(f1[0])
  xh[1][-1] += abs(f1[1])
  ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
  offsety = abs(int(f1[1]))
  offsetx = abs(int(f1[0]))
  dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)

  tmp = cv2.warpPerspective(a, xh, dsize)

  tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b  
  
  
  return tmp


def drew_matches(img1, raw_kps1, img2, raw_kps2, raw_matches):
  '''
  功能：特征点对的可视化（选取了前20个对）
  '''
  goods = []
  for m,n in raw_matches:
    if m.distance <0.75 * n.distance:
      goods.append([m])
  result = cv2.drawMatchesKnn(img1,raw_kps1, img2, raw_kps2,
                              goods[:100], None, flags = 2)
  
  return result
  

parser = argparse.ArgumentParser()
parser.add_argument('--image1', default = 's1.jpg', 
                    help = 'path to the left image')

parser.add_argument('--image2', default = 's2.jpg',
                    help = 'path to the right image')

args = vars(parser.parse_args())

left = cv2.imread(args['image1'])
right = cv2.imread(args['image2'])

left_kps1, features1 = get_the_kps_and_features(left)
right_kps2, features2 = get_the_kps_and_features(right)

raw_matches = match_keypoints(features1, features2)
result = drew_matches(left, left_kps1, right, right_kps2, raw_matches)

good_matches, H, status = find_H(left_kps1, right_kps2, raw_matches)

warp = stitch(right, left, good_matches, H, status)

rows, cols = np.where(warp[:,:,0] !=0)

min_row, max_row = min(rows), max(rows) +1
min_col, max_col = min(cols), max(cols) +1
finetune = warp[min_row:max_row,min_col:max_col,:]#去除黑色无用部分

cv2.imshow('matches',result)
cv2.imshow('stitch',finetune)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
