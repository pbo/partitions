# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FixedFormatter


def add_partitions_matrix_grid(partitions_list, axes):
    # Get partition properties
    partitions_n = partitions_list[0].n
    partitions_m = partitions_list[0].m
    p_count = len(partitions_list)

    # Calculate lexicographical grid
    tick_levels = {}  # i: level
    prev_partition = None
    for (position, partition) in enumerate(partitions_list):
        if prev_partition is None:
            tick_levels[position] = 0
        else:
            for level in range(partitions_m):
                if prev_partition[level] != partition[level]:
                    break
            tick_levels[position] = level
        prev_partition = partition

    # Sort calculated grid out
    max_level = partitions_m
    grid_template_colors = [
        (0, 0, 0, 0.8),     # 1st level
        (0, 0, 0, 0.2),     # 2nd level
        (0, 0, 0, 0.05)]    # 3rd level
    grid_template_colors += [grid_template_colors[-1]] * (max_level - len(grid_template_colors))
    grid_lines_positions = {level: [] for level in range(max_level)}
    tick_template_colors = [
        (0, 0, 0, 1.0),  # 1st level
        (0, 0, 0, 0.6),  # 2nd level
        (0, 0, 0, 0.3)]  # 3rd level
    tick_template_colors += [tick_template_colors[-1]] * (max_level - len(tick_template_colors))
    tick_length_per_level = 12
    tick_locations, tick_xtexts, tick_ytexts, tick_colors, tick_text_colors = [], [], [], [], []
    tick_lengths = []
    sublevel_position = partitions_n
    prev_level = 0
    for (position, level) in tick_levels.items():
        if level < max_level:
            # Add grid line
            grid_lines_positions[level].append(position - 0.5)

            # Save tick parameters
            tick_locations.append(position)
            tick_xtexts.append("\n".join(map(str, partitions_list[position])))
            tick_ytexts.append(" ".join(map(str, partitions_list[position])))
            #tick_xtexts.append("\n".join(map(str, partitions_list[position][level:max_level])))
            #tick_ytexts.append(" ".join(map(str, partitions_list[position][level:max_level])))
            tick_colors.append(grid_template_colors[level])
            tick_text_colors.append(tick_template_colors[level])
            tick_lengths.append(tick_length_per_level * (max_level - level - 1) + 5)

    axes.tick_params(direction = 'out',
                   top = True, left = True, right = False, bottom = False,
                   labeltop = True, labelleft = True, labelright = False, labelbottom = False)
    for level in range(max_level):
        axes.hlines(grid_lines_positions[level], -0.5, p_count-0.5, color = grid_template_colors[level], linewidth = 0.5)
        axes.vlines(grid_lines_positions[level], -0.5, p_count-0.5, color = grid_template_colors[level], linewidth = 0.5)

    def apply_ticks(axis, tick_texts):
        axis.set_ticks(tick_locations)
        axis.set_major_formatter(FixedFormatter(tick_texts))
        #for (i, tick) in enumerate(axis.get_major_ticks()):
        #   tick.set_pad(tick_lengths[i] - 1)
        for (i, label) in enumerate(axis.get_majorticklabels()):
            label.set_fontsize(8)
            label.set_color(tick_text_colors[i])
        for (i, line) in enumerate(axis.get_majorticklines()):
            line.set_color(tick_colors[i // 2])
            #line.set_markersize(tick_lengths[i // 2])
            line.set_markersize(0)

    apply_ticks(axes.xaxis, tick_xtexts)
    apply_ticks(axes.yaxis, tick_ytexts)


def plot_partition_matrix(df):
    #cmap = cm.get_cmap('viridis', df.max().max())
    plt.imshow(df)#, cmap=cmap)
    add_partitions_matrix_grid(df.columns.values, plt.gca())
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.colorbar(orientation='vertical')
    plt.show()
