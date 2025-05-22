import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.font_manager as fm

from sympy.geometry.entity import rotate

plt.rcParams['font.family'] = 'FreeSerif'

size_to_label = {
    1024:"1k", 2048:"2k",
    3072:"3k", 4096:"4k", 5120:"5k", 6144:"6k",
    7168:"7k", 8192:"8k", 9216:"9k", 10240:"10k",
    11264:"11k", 12288:"12k", 13312:"13k", 14336:"14k",
    15360:"15k", 16384:"16k", 17408:"17k", 18432:"18k",
    19456:"19k", 20480:"20k", 21504:"21k",
}

def size_to_label (size):
    if size % 1024 != 0:
        return size
    return f'{size//1024}k'


def parse_args(parser):
    parser.add_argument('--file', help='The name of the csv file to graph',
                        required=True, type=str)
    parser.add_argument('--save', help='Path to save the figure',
                        required=True, type=str)

    return parser.parse_args()


def parse_data(file_name:str, delimiter=','):
    results = dict()
    with open(file_name, 'r') as file:
        header = file.readline().strip().split(delimiter)
        lines = file.readlines()
        for line in lines:
            line_data = line.strip().split(delimiter)
            dims = []
            for dim_size in line_data[:3]:
                # Assuming the first three columns are for the dimension size
                dims.append(int(dim_size.split('.')[0]))
            dims = tuple(dims)
            for i in range(3, len(line_data)):
                if not dims in results:
                    results[dims] = dict()
                results[dims][header[i]] = float(line_data[i])

    return  results
def graph_grouped_bar(results: dict,
                      vars: tuple[str],
                      save_path: str,
                      baseline: str):
    '''
    Generate a grouped bar graph
    :param results: the results to use for creating the bar graph
    :param vars: the variables to include as part of the group
    :return: None
    '''

    x = np.arange(len(results))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.spines[['right', 'top']].set_visible(False)

    # Map each var to all its times
    var_times = dict()
    for var in vars:
        speedups = []
        for res in results.values():
            assert var in res and baseline in res
            speedups.append(round(res[var]/res[baseline], 2))
        var_times[var] = speedups

    for var, measurement in var_times.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=var, edgecolor="grey")
        # ax.bar_label(rects, padding=1, rotation=45)
        multiplier += 1


    # ax.axhline(y=1, color='black', linestyle='--')

    ax.set_title('')
    ax.set_ylabel('Speedup', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    yticks = np.arange(0, 2, step=0.25)
    ax.set_yticks(yticks)
    ax.set_xlabel('')
    ax.set_ylim(bottom=0.76, top=1.60)
    ax.set_xticks(x + width, results.keys())
    # ax.set_xticklabels([key[0] for key in results.keys()], rotation=45)
    ax.set_xticklabels([f"{size_to_label(key[0])}/{size_to_label(key[2])}" for key in results.keys()], rotation=45, fontsize=16)
    ax.legend(loc='upper left', ncols=len(vars), fontsize=18)

    ax.text(0.98, 0.95, f'Batch Size = {size_to_label(list(results.keys())[0][1])}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=18)


    plt.tight_layout()

    plt.savefig(save_path, format="png", dpi=400)
def main(args):
    results = parse_data(args.file)
    print(f'results: {results}')
    graph_grouped_bar(
        results,
        vars=('cuBLAS Dense', 'CUTLASS 2:4', 'STOICC'),
        save_path=args.save,
        baseline='cuBLAS Dense'
    )
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
