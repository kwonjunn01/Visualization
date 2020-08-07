from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt



# matplotlib 

matplotlib.rc('font', family='AppleGothic') # 

plt.rcParams['axes.unicode_minus'] = False #



# data load

digits = load_digits()



# create subplot object

fig, axes = plt.subplots(2, 5, #  allocate subplot object(2x5) to axes

                         subplot_kw={'xticks':(), 'yticks':()}) 



for ax, img in zip(axes.ravel(), digits.images): 

    ax.imshow(img)

plt.gray() # plot graph

plt.show() # plot graph

from sklearn.manifold import TSNE



# create model and learning

tsne = TSNE(random_state=0)

digits_tsne = tsne.fit_transform(digits.data)


colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]

# visualization

for i in range(len(digits.data)-1000): # To see more clearly, reduce size of data

    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), # x, y , group

             color=colors[digits.target[i]], # colours

             fontdict={'weight': 'bold', 'size':9}) # font

plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max()) # minimum to maximum

plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max()) # minimum to maximum

plt.xlabel('t-SNE characteristic0') # label of x-axis

plt.ylabel('t-SNE characterisitc1') # label of x-axis


plt.show() # plot graph
