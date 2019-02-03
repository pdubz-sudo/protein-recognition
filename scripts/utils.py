import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def filter_example(b, r, g, y):
    '''Displays filter images.'''

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
    fig.subplots_adjust(hspace=0.1)

    ax[0,0].imshow(b)
    ax[0,0].set_title('blue filter', fontsize=25)

    ax[0,1].imshow(r)
    ax[0,1].set_title('red filter', fontsize=25)

    ax[1,0].imshow(g)
    ax[1,0].set_title('green filter', fontsize=25)

    ax[1,1].imshow(y)
    ax[1,1].set_title('yellow filter', fontsize=25)
    # plt.tight_layout()
    plt.show()