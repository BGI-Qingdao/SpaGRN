import scanpy as sc
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns


def get_exp_matrix(gene_name, coordinates) -> np.array:
    gene_exp_df = coordinates.copy()
    gene_exp_df['v'] = exp_mtx[gene_name]
    df = gene_exp_df.pivot(columns='x', index='y', values='v')
    df = df.fillna(0)
    return df.to_numpy()


def get_cell_type_matrix(targer_celltype, coordinates) -> np.array:
    cell_type_df = coordinates.copy()



matrix1 = get_exp_matrix(target_genes[0], coordinates)
matrix2 = get_exp_matrix(target_genes[0], coordinates)
print(np.isnan(matrix1).any(), np.isinf(matrix1).any())
print(np.isnan(matrix2).any(), np.isinf(matrix2).any())

ssim_value, _ = ssim(matrix1, matrix2, full=True)
print(f"SSIM值: {ssim_value}")

# ax = sns.heatmap(data=df, cmap='RdYlGn', cbar=True, cbar_kws={'label': 'v'})
# plt.scatter(
# plt.show()



# fig = Figure(figsize=(5, 4), dpi=100)
# # A canvas must be manually attached to the figure (pyplot would automatically
# # do it).  This is done by instantiating the canvas with the figure as
# # argument.
# canvas = FigureCanvasAgg(fig)
#
# # your plotting here
# sc = plt.scatter(coordinates['x'],
#                  coordinates['y'],
#                  c=exp_mtx[gene_name],
#                  cmap='plasma')
# plt.gca().set_aspect('equal')
#
# canvas.draw()
# s, (width, height) = canvas.print_to_buffer()
# # Option 2a: Convert to a NumPy array.
# X = np.fromstring(s, np.uint8).reshape((height, width, 4))
# print(X.shape)
# plt.show()
#
#
#
# print(type(adata[:, gene_name].X))
# gene_expression = adata[:, gene_name].X.toarray().flatten()
# gene_expression2 = adata[:, gene_name].X#.flatten()
#
#
# x_min, y_min = coordinates.min(axis=0)
# x_max, y_max = coordinates.max(axis=0)
# grid_size_x = int(x_max - x_min) + 1
# grid_size_y = int(y_max - y_min) + 1
#
# expression_matrix = np.full((grid_size_y, grid_size_x), 0)  # Using NaN to indicate no data
# print(expression_matrix)
#
#
# # Normalize the coordinates to the grid indices
# x_indices = (coordinates[:, 0] - x_min).astype(int)
# y_indices = (coordinates[:, 1] - y_min).astype(int)
#
# expression_matrix[y_indices, x_indices] = gene_expression
#
# # Plot the resulting expression matrix
# plt.imshow(expression_matrix, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
# plt.colorbar(label=f'Expression level of {gene_name}')
# plt.title(f'2D Spatial Expression of {gene_name}')
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.show()


if __name__ == '__main__':
    