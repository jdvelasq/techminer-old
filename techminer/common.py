"""
TechMiner.common
==================================================================================================

"""
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------
# def cut_text(w):
#     if isinstance(w, (int, float)):
#         return w
#     return w if len(w) < 35 else w[:31] + '... ' + w[w.find('['):]
# #---------------------------------------------------------------------------------------------
# def chord_diagram(labels, edges, figsize=(12, 12), minval=None, R=3, n_bezier=100, dist=0.2, size=(40, 200)):

#     def bezier(p0, p1, p2, linewidth, linestyle, n_bezier=100, color='black'):

#         x0, y0 = p0
#         x1, y1 = p1
#         x2, y2 = p2

#         xb = [(1 - t)**2 * x0 + 2 * t * (1-t)*x1 + t**2 * x2 for t in np.linspace(0.0, 1.0, n_bezier)]
#         yb = [(1 - t)**2 * y0 + 2 * t * (1-t)*y1 + t**2 * y2 for t in np.linspace(0.0, 1.0, n_bezier)]

#         plt.plot(xb, yb, color=color, linewidth=linewidth, linestyle=linestyle)

#     #
#     # rutina ppal
#     #

#     plt.figure(figsize=figsize)
#     n_nodes = len(labels)

#     theta = np.linspace(0.0, 2 * np.pi, n_nodes, endpoint=False)
#     points_x = [R * np.cos(t) for t in theta]
#     points_y = [R * np.sin(t) for t in theta]


#     ## tamaños de los circulos
#     node_size = [x[(x.find('[')+1):-1] for x in labels]
#     node_size = [float(x) for x in node_size]
#     max_node_size = max(node_size)
#     min_node_size = min(node_size)
#     node_size = [size[0] + (x - min_node_size) / (max_node_size - min_node_size) * size[1] for x in node_size]


#     # dibuja los puntos sobre la circunferencia
#     plt.scatter(points_x, points_y, s=node_size, color='black', zorder=10)
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
#     plt.gca().set_aspect('equal', 'box')

#     # arcos de las relaciones
#     data = {label:(points_x[idx], points_y[idx], theta[idx]) for idx, label in enumerate(labels)}

#     ## labels
#     lbl_x = [(R+dist) * np.cos(t) for t in theta]
#     lbl_y = [(R+dist) * np.sin(t) for t in theta]
#     lbl_theta = [t / (2 * np.pi) * 360 for t in theta]
#     lbl_theta = [t - 180 if t > 180 else t for t in lbl_theta]
#     lbl_theta = [t - 180 if t > 90 else t for t in lbl_theta]

#     for txt, xt, yt, angletxt, angle  in zip(labels, lbl_x, lbl_y, lbl_theta, theta):

#         if xt >= 0:
#             ha = 'left'
#         else:
#             ha = 'right'

#         plt.text(
#             xt,
#             yt,
#             txt,
#             fontsize=10,
#             rotation=angletxt,
#             va = 'center',
#             ha = ha, # 'center'
#             rotation_mode = 'anchor',
#             backgroundcolor='white')

#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     for txt in ['bottom', 'top', 'left', 'right']:
#         plt.gca().spines[txt].set_color('white')

#     for index, r in edges.iterrows():

#         row = r['from_node']
#         col = r['to_node']
#         linewidth = r['linewidth']
#         linestyle = r['linestyle']
#         color = r['color']

#         if row != col:

#             x0, y0, a0 = data[row]
#             x2, y2, a2 = data[col]

#             angle = a0 + (a2 - a0) / 2

#             if angle > np.pi:
#                 angle_corr = angle - np.pi
#             else:
#                 angle_corr = angle

#             distance = np.abs(a2 - a0)
#             if distance > np.pi:
#                 distance = distance - np.pi
#             distance = (1.0 - 1.0 * distance / np.pi) * R / 2.5
#             x1 = distance * np.cos(angle)
#             y1 = distance * np.sin(angle)
#             x1 = 0
#             y1 = 0

#             ## dibuja los arcos
#             bezier( [x0, y0], [x1, y1], [x2, y2], linewidth=linewidth, linestyle=linestyle, color=color)

#     plt.tight_layout()
