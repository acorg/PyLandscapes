import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import to_rgb
from .landscape_utils import log2, log2_shift


def _sphere_mesh(center, radius, color, resolution=12, opacity=1.0,
                 hover_name=None, legend_name=None, legendgroup=None,
                 showlegend=False, legendrank=None):
    """Build a Mesh3d trace for a sphere at `center`.

    `hover_name` is shown in the tooltip; `legend_name` is the legend label
    (used only when `showlegend=True`).
    """
    cx, cy, cz = center
    u = np.linspace(0, 2 * np.pi, resolution * 2)
    v = np.linspace(0, np.pi, resolution)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = cx + radius * np.cos(uu) * np.sin(vv)
    y = cy + radius * np.sin(uu) * np.sin(vv)
    z = cz + radius * np.cos(vv)

    kwargs = dict(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        alphahull=0, color=color, opacity=opacity,
        flatshading=False,
        showlegend=showlegend,
        name=(legend_name if showlegend else (hover_name or "")),
        hoverinfo=("name" if hover_name else "skip"),
    )
    if legendgroup is not None:
        kwargs["legendgroup"] = legendgroup
    if legendrank is not None:
        kwargs["legendrank"] = legendrank
    
    if hover_name:
        kwargs["hovertemplate"] = f"{hover_name}<extra></extra>"
    else:
        kwargs["hoverinfo"] = "skip"
    
    trace = go.Mesh3d(**kwargs)

    # Hover should still display the per-point name even when the trace
    # contributes the group name to the legend. Achieve this with a
    # zero-size scatter overlay if the two names differ.
    return trace


def _box_wireframe(center, size, color="black", width=4, name="apex",
                   showlegend=True, legendrank=None):
    cx, cy, cz = center
    s = size / 2.0
    c = np.array([[cx + dx, cy + dy, cz + dz]
                  for dx in (-s, s) for dy in (-s, s) for dz in (-s, s)])
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
             (3,7),(4,5),(4,6),(5,7),(6,7)]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [c[i,0], c[j,0], None]
        ys += [c[i,1], c[j,1], None]
        zs += [c[i,2], c[j,2], None]
    kwargs = dict(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        name=name, hoverinfo="name", showlegend=False,
    )
    if legendrank is not None:
        kwargs["legendrank"] = legendrank
    trace1 = go.Scatter3d(**kwargs)
    
    kwargs = {"mode":"markers",
              "marker":dict(symbol="square-open", size=12, color=color,
                            line=dict(color=color, width=2)),
              "name":name, "showlegend":showlegend, "hoverinfo":"skip"}
    
    if legendrank is not None:
        kwargs["legendrank"] = legendrank

    trace2 = go.Scatter3d(
        x=[None], y=[None], z=[None], **kwargs
    )

    return [trace1, trace2]

def plot_level_set_plotly(
    H, axes, level, *,
    points=None, point_colors=None, point_names=None, point_groups=None,
    point_radii=None, apex=None, apex_box_size=None, apex_color="black",
    legend_order=None, dtick=2, opacity=0.6, colorscale="Viridis", surface_count=1,
    show=True, html_path=None, title=None, only_map=False, camera=None):
    """
    Interactive 3D level-set visualization.

    Parameters
    ----------
    axes          : list of 3 1D coordinate arrays (ij-indexed)
    H             : 3D array of scalar values
    level         : scalar iso-value, or (lo, hi) tuple for nested shells
    points        : (N, 3) array-like of point coordinates
    point_colors  : list of N color specs
    point_names   : list of N names shown in hover tooltips
    point_groups  : list of N group labels; one legend entry per unique group,
                    using the color of the first point in that group
    point_radii   : sphere radii in data units (default ~1.5% of grid extent)
    apex          : (x, y, z) apex coordinate
    apex_box_size : apex-box edge length (default = 4 * point_radius)
    apex_color    : color of the apex box edges
    legend_order  : optional list of legend entry names in the order they should
                    appear, top to bottom. Names not in the list go after,
                    in their natural order. Recognized names: any value from
                    `point_groups`, plus "apex" if an apex is drawn.
    opacity       : isosurface opacity
    dtick         : grid steps
    colorscale    : Plotly colorscale for the isosurface
    surface_count : number of nested shells (when `level` is a tuple)
    show          : open the figure
    html_path     : write to this .html file if given
    only_map      : if True only plots base map
    camera        : if provided, sets the camera position
    """
    if H.ndim != 3:
        raise ValueError(f"H must be 3D, got shape {H.shape}")

    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")

    if np.isscalar(level):
        iso_min, iso_max = float(level), float(level)
        surface_count = 1
    else:
        iso_min, iso_max = float(level[0]), float(level[1])

    ranges = [a[-1] - a[0] for a in axes]
    max_range = max(ranges)
    if point_radii is None:
        point_radii = [0.015 * max_range for _ in range(points.shape[0])]
    if apex_box_size is None:
        apex_box_size = 4 * np.mean(point_radii)

    
    if legend_order is None:
        rank_for = {}
    else:
        rank_for = {name: i for i, name in enumerate(legend_order)}

    def rank(name):
        return rank_for.get(name, 1000)

    # isosurface
    if not only_map:
      traces = [
          go.Isosurface(
              x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=H.flatten(),
              isomin=iso_min, isomax=iso_max, surface_count=surface_count,
              opacity=opacity, colorscale=colorscale,
              caps=dict(x_show=False, y_show=False, z_show=False),
              name="level set", showscale=False, showlegend=False,
          )
      ]
    else:
      traces = []

    # base map
    if points is not None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
        n = len(pts)
        if point_colors is None:
            point_colors = ["red"] * n
        elif len(point_colors) != n:
            raise ValueError(f"got {len(point_colors)} colors for {n} points")
        if point_names is None:
            point_names = [f"point {i}" for i in range(n)]
        elif len(point_names) != n:
            raise ValueError(f"got {len(point_names)} names for {n} points")
        if point_groups is not None and len(point_groups) != n:
            raise ValueError(f"got {len(point_groups)} groups for {n} points")

        # Track which group already has its legend entry assigned.
        legended = set()
        for i, (p, c, nm, pr) in enumerate(zip(pts, point_colors, point_names, point_radii)):
            grp = point_groups[i] if point_groups is not None else None
            show_in_legend = grp is not None and grp not in legended
            if show_in_legend:
                legended.add(grp)
           
          
            traces.append(_sphere_mesh(
                p, pr, c,
                hover_name=nm,
                legend_name=grp,
                legendgroup=grp,
                showlegend=show_in_legend,
                legendrank=rank(grp) if grp is not None else None,
            ))

    # apex box
    if apex is not None and not only_map:
        traces += _box_wireframe(
            apex, apex_box_size, color=apex_color,
            legendrank=rank("apex"),
            )

    fig = go.Figure(data=traces)

    fig.update_layout(
        scene=dict(
            xaxis_title="", yaxis_title="", zaxis_title="",
            aspectmode="data",
        ),
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        scene_camera=camera,
    )
    
    fig.update_layout(
    scene=dict(
        xaxis=dict(tickmode="linear", dtick=dtick, showticklabels=False),
        yaxis=dict(tickmode="linear", dtick=dtick, showticklabels=False),
        zaxis=dict(tickmode="linear", dtick=dtick, showticklabels=False),
    )
    )


    if html_path is not None:
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()

    
        
    return fig
  

def generate_grid(params, fineness, ranges, ndims):
  
  apex_height = params[ndims], 
  apex_coordinates = params[:ndims] 
  slope = params[ndims+1],
  
  apex_coordinates = np.asarray(apex_coordinates, dtype=float)
  n_dim = apex_coordinates.size
  
  if len(ranges) != n_dim:
      raise ValueError(f"got {len(ranges)} ranges for {n_dim}-D apex")

  
  axes = [np.arange(lo, hi+fineness, fineness) for (lo, hi) in ranges]
  grids = np.meshgrid(*axes, indexing="ij")
  
  sq_dist = sum((g - a) ** 2 for g, a in zip(grids, apex_coordinates))
  H = apex_height - slope * np.sqrt(sq_dist)
  
  return H,axes
  

  
def plot_titer_line(pred, target, ag_names, discrete_step, observables=None,
                    q=0.95, order=None, ax=None, linecolor="tab:red", alpha=1,
                    plot_target=True, marker='o', ci_lims=None):
  
  targets = []
  markers = []
  means = []
  
  if order is None:
    order = np.arange(len(ag_names))
  
  for ind in order:
        
    markers.append(['o' if str(x)[0]!='<' else 'v' if str(x)[0]=='<'
                    else '^' if str(x)[0]=='>' else "$?$" for x in 
                    target[target.ag_id==ind].titer])
    vals = target[target.ag_id==ind].titer.map(log2).values
    
    if discrete_step>0 and vals.size>0:
            
      mean = np.nanmean(target[target.ag_id==ind].titer.map(log2_shift))
      
      # converting thresholded to log2(x) - 1 is used for purposes of 
      # computing means but when plotting they are plotted at log2(x)
      if all(x=='v' for x in markers[-1]):
        mean += 1
      if all(x=='^' for x in markers[-1]):
        mean += -1
    elif vals.size>0:
      mean = np.nanmean(vals)
    else:
      mean = np.nan
    
    means.append(mean)
    targets.append(vals.tolist())
    
  if ax is None:
    fig,ax = plt.subplots(1, 1, figsize=(0.3*len(targets),10))
  else:
    fig = ax.get_figure()
  
  
  
  ax.plot(range(len(pred)), np.array(pred)[order], color=linecolor, zorder=-1, 
          marker=marker, markersize=5, alpha=alpha)
  if ci_lims is not None:
    ax.fill_between(range(len(pred)), np.array(ci_lims[0])[order], 
                    np.array(ci_lims[1])[order], color=linecolor, alpha=0.2)
  
  
  if observables is not None:
    top = np.quantile(log2(observables), q, axis=0)
    bottom = np.quantile(log2(observables), 1-q, axis=0)
    ax.fill_between(range(len(pred)), bottom[order], top[order], color="tab:red",
                    zorder=-1, alpha=0.1)
    
  if plot_target:
    for indt,(t,m) in enumerate(zip(targets,markers)):
      
      ax.scatter(indt, means[indt], marker='x', color="black", alpha=0.8,
                 zorder=1)
      
      if not np.isnan(means[indt]):
        ax.plot([indt, indt], [means[indt], pred[order][indt]], color="black",
                alpha=0.8, zorder=1)
      
      if len(set(m))==1:
        ax.scatter(indt*np.ones((len(t),)), t, color="black", alpha=0.3,
                   marker=m[0], zorder=0, edgecolor="black")
      else:
        m = np.array(m)
        t = np.array(t)
        for _m in set(m):
          I = np.argwhere(m==_m).flatten()
          
          ax.scatter(indt*np.ones((I.size,)), t[I], color="black", alpha=0.1,
                     marker=_m, zorder=0)
    
  ax.set_xticks(range(len(pred)))
  ax.set_xticklabels(np.array(ag_names)[order], rotation=90)
  ax.set_xlim([-1,len(pred)])
  ax.grid("on", alpha=0.1)
  ax.yaxis.set_major_locator(MultipleLocator(1))  # if you want it on y-axis too
  fig.tight_layout()

  return fig,ax


def _rgba(color, alpha):
  r,g,b = to_rgb(color)
  return f"rgba({r*255:.0f},{g*255:.0f},{b*255:.0f},{alpha})"


def plot_titer_line_plotly(pred, target, ag_names, discrete_step, observables=None,
                           q=0.95, order=None, fig=None, linecolor="tab:red", alpha=1,
                           plot_target=True, marker='o', ci_lims=None, label=None,
                           title=None, ylims=None):

  if title is None:
    title = ''
  
  sym = {'o':'circle', 'v':'triangle-down', '^':'triangle-up', "$?$":'diamond'}

  targets, markers, means = [], [], []
  if order is None:
    order = np.arange(len(ag_names))

  for ind in order:
    markers.append(['o' if str(x)[0]!='<' else 'v' if str(x)[0]=='<'
                    else '^' if str(x)[0]=='>' else "$?$" for x in
                    target[target.ag_id==ind].titer])
    vals = target[target.ag_id==ind].titer.map(log2).values

    if discrete_step>0 and vals.size>0:
      mean = np.nanmean(target[target.ag_id==ind].titer.map(log2_shift))
      if all(x=='v' for x in markers[-1]):
        mean += 1
      if all(x=='^' for x in markers[-1]):
        mean += -1
    elif vals.size>0:
      mean = np.nanmean(vals)
    else:
      mean = np.nan

    means.append(mean)
    targets.append(vals.tolist())

  x = np.arange(len(pred))
  pred_ord = np.array(pred)[order]
  fig = fig or go.Figure()

  if observables is not None:
    top = np.quantile(log2(observables), q, axis=0)[order]
    bottom = np.quantile(log2(observables), 1-q, axis=0)[order]
    fig.add_trace(go.Scatter(x=np.r_[x,x[::-1]], y=np.r_[top,bottom[::-1]], fill='toself',
                             fillcolor=_rgba("red",0.1), line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))

  pred_name = label if label is not None else "prediction"

  if ci_lims is not None:
    lo, hi = np.array(ci_lims[0])[order], np.array(ci_lims[1])[order]
    fig.add_trace(go.Scatter(x=np.r_[x,x[::-1]], y=np.r_[hi,lo[::-1]], fill='toself',
                             fillcolor=_rgba(linecolor,0.2), line=dict(width=0),
                             name=pred_name, legendgroup=pred_name, showlegend=False,
                             hoverinfo='skip'))

  if plot_target:
    target_shown = False
    for i,(t,m) in enumerate(zip(targets,markers)):
      if len(t)==0:
        continue
      one_type = len(set(m))==1
      for _m in set(m):
        I = [k for k,mm in enumerate(m) if mm==_m]
        fig.add_trace(go.Scatter(x=[i]*len(I), y=[t[k] for k in I], mode='markers',
                                 marker=dict(symbol=sym[_m], color="black",
                                            opacity=0.3 if one_type else 0.1,
                                            line=dict(color="black",width=1)),
                                 name="target", legendgroup="target",
                                 showlegend=not target_shown))
        target_shown = True
      if not np.isnan(means[i]):
        fig.add_trace(go.Scatter(x=[i,i], y=[means[i],pred_ord[i]], mode='lines',
                                 line=dict(color="black"), opacity=0.8,
                                 legendgroup="target", showlegend=False))
      fig.add_trace(go.Scatter(x=[i], y=[means[i]], mode='markers',
                               marker=dict(symbol='x', color="black"), opacity=0.8,
                               legendgroup="target", showlegend=False))

  fig.add_trace(go.Scatter(x=x, y=pred_ord, mode='lines' if marker is None else 'lines+markers',
                           line=dict(color=_rgba(linecolor,1)),
                           marker=dict(symbol=sym.get(marker,marker), size=7),
                           opacity=alpha, name=pred_name, legendgroup=pred_name,
                           showlegend=True))

  all_y = [pred_ord]
  if observables is not None:
    all_y += [top, bottom]
  if ci_lims is not None:
    all_y += [lo, hi]
  if plot_target:
    all_y.append(np.array(means, dtype=float))
    flat_targets = np.array([v for t in targets for v in t])
    if flat_targets.size:
      all_y.append(flat_targets)
  all_y = np.concatenate(all_y)
  yrange = [np.nanmin(all_y)-0.5, np.nanmax(all_y)+0.5]
  if fig.layout.yaxis.range is not None:
    old_lo,old_hi = fig.layout.yaxis.range
    yrange = [min(yrange[0],old_lo), max(yrange[1],old_hi)]
    
  if ylims is not None:
    yrange = ylims

  fig.update_layout(
    xaxis=dict(tickmode='array', tickvals=x, ticktext=np.array(ag_names)[order],
              tickangle=90, range=[-1,len(pred)], autorange=False, showgrid=True,
              gridcolor="rgba(0,0,0,0.1)", title="Antigens"),
    yaxis=dict(dtick=1, range=yrange, autorange=False, showgrid=True,
              gridcolor="rgba(0,0,0,0.1)", title="Log<sub>2</sub> Titer/10"),
    template="plotly_white", showlegend=True,
    title=title)

  return fig
