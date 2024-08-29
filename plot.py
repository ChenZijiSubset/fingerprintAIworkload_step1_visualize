import pandas as pd
import numpy as np
import gc, os, re

# Directory containing the text files
directory = '/home/ziji/fingerprintAIworkload_step1_visualize/reanalyzed_data_8cpu'

standard_columns = ['Branches', 'Branches_CPU_Using', 'All_Branches_Using', 
                    'Branch_Misses', 'Branch_Misses_CPU_Using', 
                    'Cache_References', 'Cache_References_CPU_Using', 'All_Cache_Reference_Using', 
                    'Cache_Misses', 'Cache_Misses_CPU_Using',
                    'Cycles', 'Cycles_CPU_Using',
                    'Instructions', 'Instructions_CPU_Using', 'Instructions_per_CPU_Cycle',
                    'Last_Level_Cache_Accesses', 'Last_Level_Cache_Accesses_CPU_Using',
                    'Load_Dispatches', 'Load_Dispatches_CPU_Using',
                    'Storage_Dispatches', 'Storage_Dispatches_CPU_Using',
                    'AvgMHz-', 'AvgMHz0', 'AvgMHz1', 'AvgMHz6', 'AvgMHz7', 'AvgMHz8', 'AvgMHz9', 'AvgMHz10', 'AvgMHz11', 'AvgMHz12', 'AvgMHz13',
                    'Busy-', 'Busy0', 'Busy1', 'Busy6', 'Busy7', 'Busy8', 'Busy9', 'Busy10', 'Busy11', 'Busy12', 'Busy13',
                    'BzyMHz-', 'BzyMHz0', 'BzyMHz1', 'BzyMHz6', 'BzyMHz7', 'BzyMHz8', 'BzyMHz9', 'BzyMHz10', 'BzyMHz11', 'BzyMHz12', 'BzyMHz13',
                    'C1-', 'C10', 'C11', 'C16', 'C17', 'C18', 'C19', 'C110', 'C111', 'C112', 'C113', 
                    'C2-', 'C20', 'C21', 'C26', 'C27', 'C28', 'C29', 'C210', 'C211', 'C212', 'C213',
                    'CorWatt-', 'CorWatt0', 'CorWatt1', 'CorWatt6', 'CorWatt7', 'CorWatt8', 'CorWatt9', 'CorWatt10', 'CorWatt11', 'CorWatt12', 'CorWatt13',
                    'PkgWatt-', 'PkgWatt0', 
                    'POLL-', 
                    'IRQ-', 'IRQ0', 'IRQ1', 'IRQ6', 'IRQ7', 'IRQ8', 'IRQ9', 'IRQ10', 'IRQ11', 'IRQ12', 'IRQ13', 
                    'DCCP',                
                    'CPUUtilization',
                    'rxkB/s', 'rxpck/s', 'txkB/s', 'txpck/s', '%util',
                    'TCP', 'UDP', 'UNIX', 'RAW', 'SCTP',
                    'host_to_vm1', 'vm1_to_host', 
                    'discards_completed_successfully', 'discards_merged', 'flush_requests_completed_successfully',
                    'reads_completed_successfully', 'reads_merged', 'writes_completed', 'writes_merged',
                    'time_spent_reading_(ms)', 'time_spent_writing_(ms)',
                    'sectors_read', 'sectors_written', 
                    'sectors_discarded', 'time_spent_discarding', 
                    'I/Os_currently_in_progress', 'time_spent_doing_I/Os_(ms)', 'weighted_time_spent_doing_I/Os_(ms)',
                    'time_spent_flushing',
                    ]

model_types = ['DesicionTree', 'GuassianNaiveBayes', 'KNearestNeighbors', 'MultilayerPerceptron', 'RandomForestClassifier', 'SupportVectorMachine',
                                'Resnet', 'VGG', 'ViT', 'yolo', 'BART', 'BERT']
# Helper function for natural sorting
def natural_keys(text):
    atoi = lambda text: int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Sort files by natural order
files = [f for f in os.listdir(directory) if f.endswith(".txt")]
files.sort(key=natural_keys)

# Initialize storage for interpolated data by model type and column
interpolated_data = {model_type: {column: [] for column in standard_columns} for model_type in model_types}
original_data = {model_type: {column: [] for column in standard_columns} for model_type in model_types}

# Process each file
for file in files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path, sep='\t', parse_dates=True, index_col='Timestamp', engine='python')
    
    # Determine the model type from file name
    for model_type in model_types:
        if model_type in file:
            # Process each column
            for column in standard_columns:
                column_data = df[column]
                original_data[model_type][column].append(column_data)
                # new_index = np.linspace(start=0, stop=len(column_data) - 1, num=int(median_length))
                # new_index = np.linspace(start=0, stop=len(column_data) - 1, num=485)
                # interpolated_series = np.interp(new_index, np.arange(len(column_data)), column_data)
                # print(model_type, column, interpolated_series)
                # interpolated_data[model_type][column].append(interpolated_series)
            break  # Stop checking once the model type is found

    # Clean up dataframe from memory
    del df
    print(file)
    gc.collect()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Assume 'interpolated_data' is the data already filled as discussed earlier
# data_for_plotting = []
data_for_plotting_o = []

# # Reformat the data to fit plotting needs
# for model_type, metrics in interpolated_data.items():
#     for column, data_list in metrics.items():
#         for data_array in data_list:
#             for value in data_array:  # Ensure every data point is accessed
#                 data_for_plotting.append({
#                     'Model Type': model_type,
#                     'Metric': column,
#                     'Value': value
#                 })

# Reformat the data to fit plotting needs
for model_type, metrics in original_data.items():
    print(model_type)
    for column, data_list in metrics.items():
        for data_array in data_list:
            for value in data_array:  # Ensure every data point is accessed
                data_for_plotting_o.append({
                    'Model Type': model_type,
                    'Metric': column,
                    'Value': value
                })

# Convert to DataFrame
# df = pd.DataFrame(data_for_plotting)
df_o = pd.DataFrame(data_for_plotting_o)

# Customize abbreviations for model types
custom_labels = ['DT', 'GNB', 'KNN', 'MLP', 'RF', 'SVM', 'ResNet', 'VGG', 'ViT', 'Yolo', 'BART', 'BERT']
colors = ['#c0392b', '#d35400', '#f39c12', '#2980b9', '#27ae60', '#16a085', '#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71', '#1abc9c']

custom_label_positions = range(len(custom_labels))  # Set tick positions

# metrics_plot = ['CPUUtilization', 'AvgMHz-', 'CorWatt-',                'Cache_References', 'All_Cache_Reference_Using', 'Cache_Misses']
# Ylabel = ['%', 'MHz', 'Watt', 'Number', '%', 'Number']

metrics_plot = ['CPUUtilization', 'AvgMHz-', 'Cycles',
                'Instructions', 'Branches', 'Branch_Misses', 'All_Branches_Using',
                'C1-', 'C2-',
                'POLL-', 'IRQ-',
                'Cache_References', 'All_Cache_Reference_Using', 'Cache_Misses',
                'Last_Level_Cache_Accesses', 'Load_Dispatches', 'Storage_Dispatches',
                '%util', 'TCP', 'UDP',
                'CorWatt-', 'PkgWatt-']
Ylabel = ['%', 'MHz', 'Number',
          'Number', 'Number', 'Number', '%',
          '%', '%',
          '%', 'Number',
          'Number', '%', 'Number',
          'Number', 'Number', 'Number',
          '%', 'Number', 'Number',
          'Watt', 'Watt']

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Sample DataFrame (df_o), metrics_plot, colors, custom_label_positions, custom_labels, and Ylabel need to be defined before this code

# sns.set_theme(style="whitegrid")

for i, metric in enumerate(metrics_plot):
    fig, ax = plt.subplots(figsize=(4, 2.5))  # Create a new figure and axis for each metric
    sns.violinplot(x='Model Type', y='Value', hue='Model Type', data=df_o[df_o['Metric'] == metric], palette=colors, ax=ax, legend=False)

    # Customize the plot
    plt.xticks(custom_label_positions, custom_labels, rotation=90, fontsize=16)  # Use fixed tick positions and custom labels
    ax.set_ylabel(Ylabel[i], labelpad=-2, fontsize=14)  # Set y-axis label
    ax.set_xlabel('')  # Hide x-axis label

    # Set scientific notation for y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.grid(axis='y')
    ax.tick_params(length=0)

    # Adjust the distance of y-axis tick labels, smaller pad values bring labels closer to the axis
    ax.tick_params(axis='x', which='major', pad=1, labelsize=14)  # Adjust pad value and fontsize for x-axis ticks
    ax.tick_params(axis='y', which='major', pad=1, labelsize=14)  # Adjust pad value and fontsize for y-axis ticks

    title = metric.replace('/', '')
    print(title)

    # plt.show()
    fig.savefig(f'./figures/{title}_8cpu.eps', bbox_inches='tight', pad_inches=0)
    plt.close()
