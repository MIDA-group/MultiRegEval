"""
plot utilities for the neuron project

If you use this code, please cite the first paper this was built for:
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable # plotting

def slices(slices_in,           # the 2D slices
           titles=None,         # list of titles
           cmaps=None,          # list of colormaps
           norms=None,          # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,          # option to plot the images in a grid or a single row
           width=15,            # width in in
           save=False,           # option to actually show the plot (plt.show())
           axes_off=True,
           imshow_args=None,
           save_dir='./',		# dir to save img
           save_name='sample.png'):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')
        

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots/rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i/cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.set_title(titles[i], fontsize='16')

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars and cmaps[i] is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)
    
    # clear axes that are unnecessary
    for i in range(nb_plots, col*row):
        col = np.remainder(i, cols)
        row = np.floor(i/cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows/cols*width)
    

    if save:
        plt.savefig(f'{save_dir}/{save_name}', format='png', dpi=300, bbox_inches='tight')
        plt.tight_layout()
    else:
        plt.show()

    return (fig, axs)


def flow_legend():
    """
    show quiver plot to indicate how arrows are colored in the flow() method.
    https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
    """
    ph = np.linspace(0, 2*np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.winter

    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)),  angles='xy', scale_units='xy', scale=1)
    plt.show()


def flow(slices_in,           # the 2D slices
         titles=None,         # list of titles
         cmaps=None,          # list of colormaps
         width=15,            # width in in
         img_indexing=True,   # whether to match the image view, i.e. flip y axis
         grid=False,          # option to plot the images in a grid or a single row
         show=True,            # option to actually show the plot (plt.show())
         quiver_width=None,
         scale=1):             # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots/rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i/cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][...,0], slices_in[i][...,1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale[i])
        ax.axis('equal')
    
    # clear axes that are unnecessary
    for i in range(nb_plots, col*row):
        col = np.remainder(i, cols)
        row = np.floor(i/cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows/cols*width)
    plt.tight_layout()

    if show:
        plt.show()

    return (fig, axs)


def pca(pca, x, y):
    x_mean = np.mean(x, 0)
    x_std = np.std(x, 0)

    W = pca.components_
    x_mu = W @ pca.mean_  # pca.mean_ is y_mean
    y_hat = x @ W + pca.mean_

    y_err = y_hat - y
    y_rel_err = y_err / np.maximum(0.5*(np.abs(y)+np.abs(y_hat)), np.finfo('float').eps)

    plt.figure(figsize=(15, 7))
    plt.subplot(2, 3, 1)
    plt.plot(pca.explained_variance_ratio_)
    plt.title('var %% explained')
    plt.subplot(2, 3, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim([0, 1.01])
    plt.grid()
    plt.title('cumvar explained')
    plt.subplot(2, 3, 3)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim([0.8, 1.01])
    plt.grid()
    plt.title('cumvar explained')

    plt.subplot(2, 3, 4)
    plt.plot(x_mean)
    plt.plot(x_mean + x_std, 'k')
    plt.plot(x_mean - x_std, 'k')
    plt.title('x mean across dims (sorted)')
    plt.subplot(2, 3, 5)
    plt.hist(y_rel_err.flat, 100)
    plt.title('y rel err histogram')
    plt.subplot(2, 3, 6)
    plt.imshow(W @ np.transpose(W), cmap=plt.get_cmap('gray'))
    plt.colorbar()
    plt.title('W * W\'')
    plt.show()
