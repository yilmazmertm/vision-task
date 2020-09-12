from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import utils

image = cv2.imread('./Small_area.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis('off')
plt.imshow(image)

image = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape)

clt = KMeans(n_clusters = 3)
clt.fit(image)

hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()