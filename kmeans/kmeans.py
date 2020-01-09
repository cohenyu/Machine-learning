import numpy as np
from numpy import floor
from init_centroids import init_centroids
from scipy.misc import imread


# this class represent a cluster
class Cluster:
    numOfPixels = 0
    redSum = 0
    greenSum = 0
    blueSum = 0


# this function returns the distance between pixel to centroid
def distance(pixel, centroid):
    return np.linalg.norm(pixel - centroid)


# The function returns the index of the nearest centroid to the given pixel
def classify(centroids, my_pixel):
    min_distance = distance(my_pixel, centroids[0])
    min_index = 0
    for index, centroid in enumerate(centroids):
        cur_distance = distance(my_pixel, centroid)
        # If we find a closer centroid, we will update the index and distance
        if cur_distance < min_distance:
            min_distance = cur_distance
            min_index = index
    return min_index


# The function adds the pixel to the given cluster.
def assign(pixel, cluster):
    cluster.redSum += pixel[0]
    cluster.greenSum += pixel[1]
    cluster.blueSum += pixel[2]
    cluster.numOfPixels += 1


# The function creates an array of clusters. (k-size)
def create_array(k):
    array = []
    for i in range(k):
        array.append(Cluster())
    return array


def main():
    path = 'dog.jpeg'
    # Iterate on the number of centroids.
    for k in [2, 4, 8, 16]:
        # data preparation (loading, normalizing, reshaping)
        A = imread(path)
        A = A.astype(float) / 255.
        img_size = A.shape
        X = A.reshape(img_size[0] * img_size[1], img_size[2])
        print('k=' + repr(k) + ':')
        centroids = init_centroids(X, k)

        for i in range(11):
            cluster_array = create_array(k)
            print('iter ' + repr(i) + ': ', end="")

            array = []
            for centroid in centroids:
                array2 = []
                for value in centroid:
                    strVal = str(floor(value * 100) / 100)
                    if strVal == '0.0':
                        array2.append('0.')
                    else:
                        array2.append(strVal)
                array.append(array2)

            # Printing the values ​​of the centroids.
            y = ", ".join(map(str, array))
            for centroid in y:
                x = ", ".join(centroid)
                print(x.replace("'", ""), end="")
            print("")

            # pixel classification to a centroid and adding it to the appropriate cluster.
            for pixel in X:
                index = classify(centroids, pixel)
                assign(pixel, cluster_array[index])

            # Update a new location for each centroid to the average of all the pixels in the same cluster.
            for index, cluster in enumerate(cluster_array):
                if cluster.numOfPixels != 0:
                    centroids[index][0] = cluster.redSum / cluster.numOfPixels
                    centroids[index][1] = cluster.greenSum / cluster.numOfPixels
                    centroids[index][2] = cluster.blueSum / cluster.numOfPixels

        # Update the pixels in their new color
        for pixel in X:
            index = classify(centroids, pixel)
            centroid = centroids[index]
            pixel[0] = centroid[0]
            pixel[1] = centroid[1]
            pixel[2] = centroid[2]


if __name__ == "__main__":
    main()