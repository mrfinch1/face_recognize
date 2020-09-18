from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1_path = "test/a1.jpg" #yüzünü tanımasını isteyeceğiniz kişi
img2_path = "test/a2.jpg" #yüzünü tanıtığınız 1.kişi
img3_path = "test/a3.jpg" #yüzünü tanıttığınız 2.kişi
img4_path = "test/a4.jpg" #yüzünü tanıttığınız 3.kişi
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)
img4 = cv2.imread(img4_path)
plt.imshow(img1[:,:,::-1])
plt.show()

plt.imshow(img2[:,:,::-1])
plt.show()

plt.imshow(img3[:,:,::-1])
plt.show()

plt.imshow(img3[:,:,::-1])
plt.show()
resp = DeepFace.verify(img1,img2)
resp1 = DeepFace.verify(img1,img3)

print(resp["verified"])
if resp["verified"] == 1:
  print("Fotoğraftaki kişi Kemal Sunal")
elif resp1["verified"] == 1:
  print("Fotoğraftaki kişi elon musk")
else :
  print("Sonuç bulunamadı")
