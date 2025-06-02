#!/usr/bin/env python3
"""Generate architecture diagram for YOLOX finetuning system"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Define component positions
components = {
    'CLI': (2, 9, 2, 1),
    'Config': (5, 9, 2, 1),
    'Monitor': (8, 9, 2, 1),
    'Export': (11, 9, 2, 1),
    
    'Trainer': (7, 6.5, 3, 1.5),
    
    'Model Factory': (2, 4, 2.5, 1),
    'Data Pipeline': (5, 4, 2.5, 1),
    'Device Manager': (8, 4, 2.5, 1),
    'Checkpoint': (11, 4, 2.5, 1),
    
    'Formats': (2, 1.5, 2, 1),
    'Augmentation': (4.5, 1.5, 2, 1),
    'Validation': (7, 1.5, 2, 1),
    'Metrics': (9.5, 1.5, 2, 1),
    'Visualizer': (12, 1.5, 2, 1),
}

# Color scheme
colors = {
    'interface': '#4CAF50',
    'core': '#2196F3',
    'data': '#FF9800',
    'utils': '#9C27B0',
    'monitoring': '#F44336'
}

# Draw components
for name, (x, y, w, h) in components.items():
    if name in ['CLI', 'Config', 'Monitor', 'Export']:
        color = colors['interface']
    elif name == 'Trainer':
        color = colors['core']
    elif name in ['Data Pipeline', 'Formats', 'Augmentation', 'Validation']:
        color = colors['data']
    elif name in ['Model Factory', 'Device Manager', 'Checkpoint']:
        color = colors['utils']
    else:
        color = colors['monitoring']
    
    box = FancyBboxPatch((x, y), w, h, 
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='black',
                         alpha=0.8,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, name, 
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            color='white' if name != 'Config' else 'black')

# Draw connections
connections = [
    # CLI to components
    ('CLI', 'Trainer', 'solid'),
    ('Config', 'Trainer', 'solid'),
    ('Monitor', 'Trainer', 'dashed'),
    ('Export', 'Model Factory', 'dashed'),
    
    # Trainer to core components
    ('Trainer', 'Model Factory', 'solid'),
    ('Trainer', 'Data Pipeline', 'solid'),
    ('Trainer', 'Device Manager', 'solid'),
    ('Trainer', 'Checkpoint', 'solid'),
    
    # Sub-components
    ('Data Pipeline', 'Formats', 'solid'),
    ('Data Pipeline', 'Augmentation', 'solid'),
    ('Data Pipeline', 'Validation', 'solid'),
    ('Trainer', 'Metrics', 'dashed'),
    ('Trainer', 'Visualizer', 'dashed'),
]

for start, end, style in connections:
    start_pos = components[start]
    end_pos = components[end]
    
    # Calculate connection points
    start_x = start_pos[0] + start_pos[2]/2
    start_y = start_pos[1]
    end_x = end_pos[0] + end_pos[2]/2
    end_y = end_pos[1] + end_pos[3]
    
    # Draw arrow
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), 
                           "data", "data",
                           arrowstyle="->",
                           shrinkA=5, shrinkB=5,
                           mutation_scale=20,
                           fc="black",
                           linestyle=style,
                           linewidth=1.5,
                           alpha=0.6)
    ax.add_artist(arrow)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=colors['interface'], label='User Interface'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['core'], label='Core System'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['data'], label='Data Components'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['utils'], label='Utilities'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['monitoring'], label='Monitoring'),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

# Add title and labels
ax.set_title('YOLOX Finetuning System Architecture', fontsize=16, fontweight='bold', pad=20)
ax.text(7, 0.5, 'Data Flow', ha='center', fontsize=12, style='italic')

# Set axis limits and remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Add annotations
ax.text(7, 8, '↓ User Commands', ha='center', fontsize=10, style='italic')
ax.text(7, 5.5, '↓ Training Process', ha='center', fontsize=10, style='italic')
ax.text(7, 3, '↓ Core Components', ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('finetuning_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('finetuning_architecture.pdf', bbox_inches='tight')
print("Architecture diagram saved as 'finetuning_architecture.png' and 'finetuning_architecture.pdf'")