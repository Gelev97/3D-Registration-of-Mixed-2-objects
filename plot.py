import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# get color from http://colormind.io/
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

sns.set(font='Franklin Gothic Book',
        rc={
    'axes.axisbelow': False,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',
    'axes.grid': False,
    'axes.labelcolor': 'dimgrey',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',
    'xtick.bottom': False,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.left': False,
    'ytick.right': False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def autolabel(bar_plot, bar_label,ax):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)

def plotComparison(filename_arr,x,x_label):
    data = [[],[]]
    for filename in filename_arr:
        result = np.load(filename)
        total_err = reject_outliers(result["total_err"])
        total_time = reject_outliers(result["total_time"])
        total_err_avg = np.sum(total_err)/total_err.shape[0]
        total_time_avg = np.sum(total_time)/total_time.shape[0]
        data[0].append(total_err_avg)
        data[1].append(total_time_avg)

    fig, ax = plt.subplots()
    bar_plot = plt.bar(x,data[0],color=(CB91_Blue, CB91_Green, CB91_Pink, CB91_Purple))
    autolabel(bar_plot, np.round(data[0],3), ax)
    # autolabel(bar_plot, np.round(data[0],12), ax)
    plt.xlabel(x_label)
    plt.ylabel('Avg reprojection error (L1 norm)')
    plt.show()
    fig, ax = plt.subplots()
    bar_plot = plt.bar(x,data[1],color=(CB91_Blue, CB91_Green, CB91_Pink, CB91_Purple))
    autolabel(bar_plot, np.round(data[1],2), ax)
    plt.xlabel(x_label)
    plt.ylabel('Avg runtime(s)')
    plt.show()

def plotResult(filename):
    result = np.load(filename)
    total_err = result["total_err"]
    total_time = result["total_time"]
    num_bins = 20
    n, bins, patches = plt.hist(total_err, num_bins,color=CB91_Blue)
    plt.xlabel('reprojection error (L1 norm)')
    plt.ylabel('test case number')
    plt.show()
    n, bins, patches = plt.hist(total_time, num_bins,color=CB91_Green)
    plt.xlabel('runtime(s)')
    plt.ylabel('test case number')  
    plt.show()

if __name__ == "__main__": 
    plotResult('./results/object2_test100_total400.npz')
    plotResult('./results/object3_test100_total400.npz')
    plotResult('./results/object4_test100_total400.npz')
    plotResult('./results/object5_test100_total400.npz')
    plotResult('./results/object2_test100_total4096.npz')

    filename_arr = ['./results/object2_test100_total400.npz',
                './results/object3_test100_total400.npz',
                './results/object4_test100_total400.npz',
                './results/object5_test100_total400.npz',
                "./results/object2_test100_total4096.npz"]
    plotComparison(filename_arr, ["2 objects(400 points)", "3 objects", "4 objects", "5 objects", "2 objects(4096 points)"],"object number")

    plotResult('./results/object2_test100_total400_noise1.npz')
    plotResult('./results/object2_test100_total400_noise2.npz')
    plotResult('./results/object2_test100_total400_noise3.npz')
    plotResult('./results/object2_test100_total400_noise4.npz')
    plotResult('./results/object2_test100_total400_noise5.npz')

    filename_arr = ['./results/object2_test100_total400_noise1.npz',
                    './results/object2_test100_total400_noise2.npz',
                    './results/object2_test100_total400_noise3.npz',
                    './results/object2_test100_total400_noise4.npz',
                    './results/object2_test100_total400_noise5.npz']
    plotComparison(filename_arr, ["0.01", "0.02", "0.03", "0.04","0.05"],"gaussian noise level")
