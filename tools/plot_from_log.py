import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines


def logfile_to_df(logfile):
    """Generate Dataframe of the form 

                    _type  epoch      eta  gpu_mem iter      loss        lr  time_diff   top1_err   top5_err   RAM  min_top1_err  min_top5_err
        51     train_epoch      1  1:58:44  4.44 GB  NaN  4.086279  0.010047   0.071294  93.009709  78.082524  6.20           NaN           NaN
        103    train_epoch      2  0:50:27  4.44 GB  NaN  3.671082  0.010094   0.030327  90.461165  70.436893  6.07           NaN           NaN

        from the json_stats outputted to the console while training.

    Arguments:
        logfile {string} -- relative or absolute path to the logfile (stdout.log)

    Returns:
        Dataframe
    """
    # Read Logfile
    df_log = pd.read_csv(logfile, header=None, names=['row'],
                         sep="non_existing_seperator", engine="python")

    # Remove all rows that dont contain json:
    df_log = df_log[df_log["row"].str.contains("json_stats: ")]

    # Remove the beginning (non-json) part of the row:
    df_log['json'] = df_log['row'].str.split('json_stats: ').str[1]
    # Format json column as json dict:
    df_jsoncol = df_log['json'].apply(json.loads)
    # Convert json dict to pandas columns.
    df = pd.io.json.json_normalize(df_jsoncol)

    # Remove the total nr. of epochs from the epoch field:
    df['epoch'] = df['epoch'].str.split('/').str[0].astype(int)
    # Remove the total amount of RAM and the unit from the RAM field:
    df['RAM'] = df['RAM'].str.split('/').str[0].astype(float)

    return df


def plot_errorrates(df, showTitle=True, legendTitle="Error rate during training:"):
    fig = plt.figure(figsize=(9, 3.5), num='plot_errorrates')
    ax = fig.add_subplot(1, 1, 1)
    if(showTitle):
        ax.set_title("Error Rates during Training Epochs")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot Top-1 and Top 5-Error for both train and val epochs (in inverse z-order)
    params = dict(kind='line', x='epoch', linewidth=2, ax=ax)
    df[df._type == "train_epoch"].plot(y='top5_err', label='top-5 (train)',
                                       linestyle='dashed', color='#ffc107', **params)
    df[df._type == "train_epoch"].plot(y='top1_err', label='top-1 (train)',
                                       color='#ffc107', **params)

    df[df._type == "val_epoch"].plot(y='top5_err', label='top-5 (val)',
                                     linestyle='dashed', color='#17a2b8', **params)
    df[df._type == "val_epoch"].plot(y='top1_err', label='top-1 (val)',
                                     color='#17a2b8', **params)

    # Show Ticks between min and max nr. of epochs
    ax.set_xlim((1, df['epoch'].max()))
    ax.set_xticks(np.concatenate(
        (np.arange(20, df['epoch'].max(), 20.0), [1, df['epoch'].max()])))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    # Show Ticks between 0 and 100%
    ax.set_ylim((0, 100))
    ax.set_yticks(np.arange(0, 100+1, 10.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # Show a Grid based on the ticks in the background
    ax.grid(which='both', color='#6c757d', linestyle='-', linewidth=0.2)
    # Hide epoch label (self-explanatory when title is shown)
    ax.xaxis.label.set_visible(False)
    # reverse order the labels (to represent z-order) and handles and add a spacing in between
    current_handles, current_labels = ax.get_legend_handles_labels()
    reversed_handles = list(reversed(current_handles))
    #reversed_handles.insert(int(len(current_handles)/2),
    #                        mlines.Line2D([], [], linestyle=''))
    reversed_labels = list(reversed(current_labels))
    #reversed_labels.insert(int(len(current_labels)/2), '')
    # call .legend() with the new values
    # Increase legend size and add title for better readabilty
    leg = ax.legend(reversed_handles, reversed_labels, loc='upper right',
              borderpad=1, labelspacing=0.6, handlelength=5, title=legendTitle)
    leg._legend_box.align = "left"
    return fig


def plot_learningrate(df, showTitle=True):
    fig = plt.figure(figsize=(4, 2), num='plot_learningrate')
    ax = fig.add_subplot(1, 1, 1)
    if(showTitle):
        ax.set_title("Learning Rate used in Training Epochs")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot Learning Rate for all train epochs
    df[df._type == "train_epoch"].plot(kind='line', x='epoch', y='lr',
                                       color='#343a40', linewidth=2, ax=ax)

    # Show Ticks between min and max nr. of epochs
    ax.set_xlim((1, df['epoch'].max()))
    ax.set_xticks(np.concatenate(
        (np.arange(50, df['epoch'].max(), 50.0), [1, df['epoch'].max()])))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    # Show Ticks between min and max learning rate
    ax.set_ylim((df['lr'].min(), df['lr'].max()))
    ax.set_yticks([df['lr'].min(), df['lr'].max(), df['lr'].iloc[0]])
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.001))
    # Hide epoch label (self-explanatory when title is shown)
    ax.xaxis.label.set_visible(False)
    # Remove legend for better readabilty
    ax.get_legend().remove()
    return fig

def plot_accuracies(showTitle=True, top5=(0,0,0), top1=(0,0,0), legendTitle="Accuracies:"):

    labels = ['train','val','test']
    x = np.arange(len(labels))  # the label locations
    width = 0.9  # the width of the bars

    fig = plt.figure(figsize=(4, 2), num='plot_accuracies')
    ax = fig.add_subplot(1, 1, 1)
    if(showTitle):
        ax.set_title("Model Accuracy for different splits")

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Show Ticks between 0 and 100%
    ax.set_xlim((30, 100))
    ax.set_xticks(np.arange(40, 100+1, 20.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    
    # Show Dataset Split Labels on y axis
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
        
    # Plot the Accuracies as Horizontal Bars
    rects1 = ax.barh(x, top5, width, label='top-5', color='#17a2b8', zorder=3)
    rects2 = ax.barh(x, top1, width, label='top-1', color='#ffc107', zorder=4)
    
    # Show a Grid based on the ticks in the background
    ax.grid(which='both', axis='x', color='#6c757d', linestyle='-', linewidth=0.2, zorder=0)
    
    # reverse order the labels (to represent z-order)
    current_handles, current_labels = ax.get_legend_handles_labels()
    reversed_handles = list(reversed(current_handles))
    reversed_labels = list(reversed(current_labels))
    
    # Plot legend and add title for better readabilty
    leg = ax.legend(reversed_handles, reversed_labels, loc='lower left', ncol=2, title=legendTitle)
    leg._legend_box.align = "left"

    def autolabel(rects, offset=(0,0), color='red', text=''):
        """Attach a text label for each bar in *rects*, displaying its width."""
        for rect in rects:
            width = rect.get_width()
            ax.text(width+offset[0],
                    rect.get_y() + (rect.get_height() / 2) + offset[1],
                    text+'{}%'.format(width),
                    bbox=dict(boxstyle='square,pad=0.1', fc=color, ec='none'),
                    ha='center', va='bottom', zorder=5)


    autolabel(rects1, offset=(-7.5,-0.35), color='#17a2b8')
    autolabel(rects2, offset=(-7.5,0.05), color='#ffc107')

    return fig


if __name__ == "__main__":
    df = logfile_to_df(logfile="output/retrain1/retrainresults/stdout_retrain1.log")
    print(df[df._type == "train_epoch"])
    print(df[df._type == "val_epoch"])
    
    plot_errorrates(df, showTitle=False, legendTitle="Fehlerrate in Trainings-Epoche:")
    plt.savefig('output/plot_errorrates.pdf', bbox_inches = 'tight', pad_inches = 0)

    #plot_learningrate(df, showTitle=False)
    #plt.savefig('output/plot_learningrate.pdf', bbox_inches = 'tight', pad_inches = 0)

    #plot_accuracies(top5 = (100-0.44, 100-19.09, 100-19.78), top1 = (100-0.58, 100-38.19, 100-41.19), showTitle=False, legendTitle="Genauigkeit:")
    plot_accuracies(top5 = (100-0.44, 100-19.02, 100-19.47), top1 = (100-0.68, 100-38.34, 100-40.19), showTitle=False, legendTitle="Genauigkeit:")
    plt.savefig('output/plot_accuracies.pdf', bbox_inches = 'tight', pad_inches = 0)

    plt.show()
