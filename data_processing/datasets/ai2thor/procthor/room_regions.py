"""
Room region detection from wall/floor/ceiling segments.

Deduplicates walls, builds a planar graph from segment intersections,
finds minimal faces as room regions, and assigns walls/floors/ceilings to rooms.
"""

import numpy as np
from matplotlib.path import Path


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TOL = 1e-6
DECIMALS = 4
MERGE_DIST = 0.08
MERGE_DIST_SQ = MERGE_DIST * MERGE_DIST
ON_BOUNDARY_SQ = MERGE_DIST * MERGE_DIST * 2.25
WALL_ROOM_ASSIGN_DIST_SQ = (MERGE_DIST * 2) ** 2


# -----------------------------------------------------------------------------
# Geometry helpers (segment/point math)
# -----------------------------------------------------------------------------
def _segment_intersection(a, b, c, d):
    """Intersection of segments (a,b) and (c,d). Returns point if interior or endpoint intersection, else None."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None  # parallel or coincident
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def _segments_overlap_collinear(a, b, u, v, tol=1e-9):
    """True if collinear segments (a,b) and (u,v) overlap or touch (within tol)."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    ux, uy = u[0], u[1]
    vx, vy = v[0], v[1]
    if abs(bx - ax) < 1e-12:  # vertical
        if abs(ux - ax) > tol:
            return False
        s1_min, s1_max = min(ay, by), max(ay, by)
        s2_min, s2_max = min(uy, vy), max(uy, vy)
        return s1_max >= s2_min - tol and s2_max >= s1_min - tol
    # horizontal
    if abs(by - ay) > 1e-12:
        return False
    if abs(uy - ay) > tol:
        return False
    s1_min, s1_max = min(ax, bx), max(ax, bx)
    s2_min, s2_max = min(ux, vx), max(ux, vx)
    return s1_max >= s2_min - tol and s2_max >= s1_min - tol


def _point_on_segment(p, a, b, tol=1e-9):
    """True if p lies on segment (a,b) (including endpoints) within tol."""
    px, py = p
    ax, ay = a
    bx, by = b
    seg_len_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if seg_len_sq < 1e-16:
        return (px - ax) ** 2 + (py - ay) ** 2 <= tol * tol
    perp = abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / (seg_len_sq ** 0.5)
    if perp > tol:
        return False
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / seg_len_sq
    return -tol <= t <= 1 + tol


def _point_to_segment_dist_sq(p, a, b):
    """Squared distance from point p to line segment (a,b)."""
    px, py = p
    ax, ay = a
    bx, by = b
    seg_len_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if seg_len_sq < 1e-16:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def _snap_points(points, merge_dist_sq):
    """Merge points within merge_dist; return unique representative points and mapping from index to rep."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    def dist2(i, j):
        a, b = points[i], points[j]
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    for i in range(n):
        for j in range(i + 1, n):
            if dist2(i, j) <= merge_dist_sq:
                union(i, j)

    comps = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(points[i])

    unique = []
    idx_to_rep = {}
    for r, pts in comps.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        rep = (round(cx, 4), round(cy, 4))
        unique.append(rep)
        for i in range(n):
            if find(i) == r:
                idx_to_rep[i] = rep
    return unique, idx_to_rep


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _canon(p):
    return (round(float(p[0]), DECIMALS), round(float(p[1]), DECIMALS))


def _seg_key(a, b):
    return (min(a, b), max(a, b))


# -----------------------------------------------------------------------------
# Step 1: Deduplicate wall segments (merge overlapping collinear)
# -----------------------------------------------------------------------------
def _seg_direction(a, b):
    dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
    if dx < 1e-8:
        return ("v", a[0])
    if dy < 1e-8:
        return ("h", a[1])
    return ("d", (a[0], a[1], b[0] - a[0], b[1] - a[1]))


def _flatten_wall_segments(wall_segments_dict):
    """Flatten wall segments with geom name: list of (a, b, geom_name)."""
    raw = []
    for geom_name, lines in wall_segments_dict.items():
        for (p, q) in lines:
            a, b = _canon(p), _canon(q)
            if _dist2(a, b) > 1e-12:
                raw.append((a, b, geom_name))
    return raw


def _deduplicate_wall_segments(raw):
    """
    Merge overlapping collinear segments, then deduplicate by canonical key.
    Returns (segments, segment_geom_names, segment_geom_with_segments).
    """
    merged_segs = []
    used = [False] * len(raw)
    for i, (a, b, gname) in enumerate(raw):
        if used[i]:
            continue
        seg = (a, b)
        geom_with_segments = [(gname, (a, b))]
        used[i] = True
        while True:
            extended = False
            d = _seg_direction(seg[0], seg[1])
            for j in range(len(raw)):
                if used[j]:
                    continue
                c, d2, gname_j = raw[j][0], raw[j][1], raw[j][2]
                if _seg_direction(c, d2) != d:
                    continue
                a0, b0 = seg
                if d[0] == "v":
                    x = a0[0]
                    if abs(c[0] - x) > MERGE_DIST or abs(d2[0] - x) > MERGE_DIST:
                        continue
                    y_min = min(a0[1], b0[1], c[1], d2[1])
                    y_max = max(a0[1], b0[1], c[1], d2[1])
                    if max(a0[1], b0[1]) < min(c[1], d2[1]) - MERGE_DIST or max(c[1], d2[1]) < min(a0[1], b0[1]) - MERGE_DIST:
                        continue
                    seg = ((x, y_min), (x, y_max))
                    geom_with_segments.append((gname_j, (c, d2)))
                    used[j] = True
                    extended = True
                    break
                elif d[0] == "h":
                    y = a0[1]
                    if abs(c[1] - y) > MERGE_DIST or abs(d2[1] - y) > MERGE_DIST:
                        continue
                    x_min = min(a0[0], b0[0], c[0], d2[0])
                    x_max = max(a0[0], b0[0], c[0], d2[0])
                    if max(a0[0], b0[0]) < min(c[0], d2[0]) - MERGE_DIST or max(c[0], d2[0]) < min(a0[0], b0[0]) - MERGE_DIST:
                        continue
                    seg = ((x_min, y), (x_max, y))
                    geom_with_segments.append((gname_j, (c, d2)))
                    used[j] = True
                    extended = True
                    break
            if not extended:
                break
        merged_segs.append((seg, geom_with_segments))

    seen = {}
    segments = []
    segment_geom_names = []
    segment_geom_with_segments = []
    for (a, b), geom_with_segments in merged_segs:
        k = _seg_key(a, b)
        if _dist2(a, b) <= 1e-12:
            continue
        gnames_set = {g for g, _ in geom_with_segments}
        if k not in seen:
            seen[k] = len(segments)
            segments.append((a, b))
            segment_geom_names.append(set(gnames_set))
            segment_geom_with_segments.append(list(geom_with_segments))
        else:
            idx = seen[k]
            segment_geom_names[idx].update(gnames_set)
            segment_geom_with_segments[idx].extend(geom_with_segments)
    return segments, segment_geom_names, segment_geom_with_segments


# -----------------------------------------------------------------------------
# Step 2: Vertices and edges
# -----------------------------------------------------------------------------
def _collect_vertices(segments):
    """Vertices = segment endpoints + pairwise intersections, snapped and deduped."""
    points = []
    for (a, b) in segments:
        points.append(a)
        points.append(b)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a, b = segments[i]
            c, d = segments[j]
            inter = _segment_intersection(a, b, c, d)
            if inter is not None:
                points.append(inter)
    points = [_canon(p) for p in points]
    vertices, _ = _snap_points(points, MERGE_DIST_SQ)
    return list(dict.fromkeys(vertices))


def _param(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    if abs(bx - ax) >= abs(by - ay):
        return (px - ax) / (bx - ax) if (bx - ax) != 0 else 0.0
    return (py - ay) / (by - ay) if (by - ay) != 0 else 0.0


def _build_edges_with_geoms(segments, segment_geom_names, segment_geom_with_segments, vertices):
    """
    For each segment, collect vertices on it, sort, emit sub-edges with geom names.
    Returns (edges, edge_geom_names, edge_geom_names_unmerged).
    """
    edges_with_geoms = []
    for seg_idx, (a, b) in enumerate(segments):
        gnames_merged = segment_geom_names[seg_idx]
        geom_with_segments = segment_geom_with_segments[seg_idx]
        on_seg = [v for v in vertices if _point_on_segment(v, a, b, MERGE_DIST)]
        if len(on_seg) < 2:
            continue
        on_seg.sort(key=lambda p: _param(p, a, b))
        out = [on_seg[0]]
        for p in on_seg[1:]:
            if _dist2(p, out[-1]) > MERGE_DIST_SQ:
                out.append(p)
        for i in range(len(out) - 1):
            u, v = out[i], out[i + 1]
            if _dist2(u, v) <= 1e-12:
                continue
            gnames_overlap = [
                gname for gname, orig_seg in geom_with_segments
                if _segments_overlap_collinear(orig_seg[0], orig_seg[1], u, v, MERGE_DIST)
            ]
            edges_with_geoms.append((u, v, gnames_merged, gnames_overlap))

    seen_edges = set()
    unique_edges = []
    edge_geom_names = []
    edge_geom_names_unmerged = []
    edge_key_to_id = {}
    for (u, v, gnames_merged, gnames_overlap) in edges_with_geoms:
        k = _seg_key(u, v)
        if k not in seen_edges:
            seen_edges.add(k)
            edge_key_to_id[k] = len(unique_edges)
            unique_edges.append((u, v))
            edge_geom_names.append(set(gnames_merged))
            edge_geom_names_unmerged.append(list(gnames_overlap))
        else:
            edge_geom_names[edge_key_to_id[k]].update(gnames_merged)
            edge_geom_names_unmerged[edge_key_to_id[k]].extend(gnames_overlap)
    return unique_edges, edge_geom_names, edge_geom_names_unmerged


# -----------------------------------------------------------------------------
# Step 3: Planar graph and minimal faces (room regions)
# -----------------------------------------------------------------------------
def _find_minimal_faces(edges, vertices):
    """
    Build adjacency sorted by angle, trace faces by following right-hand boundary,
    normalize and deduplicate. Keep only minimal faces:
    - Every consecutive pair of vertices in the polygon must be an actual graph edge.
    - No vertex of the graph lies strictly inside the face.
    - No other face (closed loop) lies entirely inside this face.
    """
    # Canonical edge set for strict "polygon boundary = graph edges" check
    edge_set = {_seg_key(u, v) for (u, v) in edges}

    def polygon_edges_are_in_graph(poly):
        """True iff every consecutive pair (poly[i], poly[i+1]) is an edge in the graph."""
        n = len(poly)
        for i in range(n):
            u, v = poly[i], poly[(i + 1) % n]
            if _seg_key(u, v) not in edge_set:
                return False
        return True

    adj = {}
    for i, (u, v) in enumerate(edges):
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))

    def angle_from(ref, p):
        return np.arctan2(p[1] - ref[1], p[0] - ref[0])

    for u in adj:
        adj[u].sort(key=lambda x: angle_from(u, x[0]))

    def next_edge(u, v, edge_id):
        neighbors = adj.get(v, [])
        if not neighbors:
            return None, None, None
        idx = next((i for i, (w, e) in enumerate(neighbors) if w == u), None)
        if idx is None:
            return None, None, None
        next_idx = (idx - 1) % len(neighbors)
        w, eid = neighbors[next_idx]
        return v, w, eid

    used_directed = set()
    faces = []
    for u, v in edges:
        for (a, b) in [(u, v), (v, u)]:
            key = (a, b)
            if key in used_directed:
                continue
            path = [a, b]
            used_directed.add(key)
            cur, nxt = a, b
            eid = next((i for i, (x, y) in enumerate(edges) if (x, y) == (cur, nxt) or (x, y) == (nxt, cur)), None)
            while True:
                cur, nxt, eid = next_edge(cur, nxt, eid)
                if cur is None:
                    break
                if (cur, nxt) in used_directed:
                    break
                used_directed.add((cur, nxt))
                path.append(nxt)
                if nxt == path[0] and len(path) >= 3:
                    faces.append(path[:-1])
                    break

    def normalize_face(cycle):
        if not cycle or len(cycle) < 3:
            return None
        idx = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
        rot = cycle[idx:] + cycle[:idx]
        if (rot[1][0], rot[1][1]) <= (rot[-1][0], rot[-1][1]):
            return tuple(tuple(p) for p in rot)
        return tuple(tuple(p) for p in [rot[0]] + list(rot[1:])[::-1])

    seen_faces = set()
    all_faces = []
    for f in faces:
        key = normalize_face(f)
        if key is not None and key not in seen_faces:
            seen_faces.add(key)
            all_faces.append(list(key))

    # Keep only faces whose boundary is exactly graph edges (no "diagonal" between non-adjacent vertices)
    edge_valid_faces = [p for p in all_faces if len(p) >= 3 and polygon_edges_are_in_graph(p)]

    all_vertices_set = set(vertices)

    def on_boundary(poly, p):
        for q in poly:
            if _dist2(q, p) <= ON_BOUNDARY_SQ:
                return True
        return False

    def polygon_centroid(poly):
        n = len(poly)
        sx = sum(p[0] for p in poly) / n
        sy = sum(p[1] for p in poly) / n
        return (sx, sy)

    def face_contains_other_face(outer_poly, inner_poly):
        """True if inner_poly is entirely inside outer_poly (and not equal)."""
        if outer_poly is inner_poly or len(inner_poly) < 3:
            return False
        path_outer = Path(np.asarray(outer_poly))
        for p in inner_poly:
            if not on_boundary(outer_poly, p) and not path_outer.contains_point(p, radius=-1e-6):
                return False
        # All vertices of inner are on or inside outer; ensure at least one strictly inside
        cx, cy = polygon_centroid(inner_poly)
        if path_outer.contains_point((cx, cy), radius=-1e-6) and not on_boundary(outer_poly, (cx, cy)):
            return True
        return False

    # Minimal = no vertex strictly inside AND no other face (closed loop) entirely inside
    room_regions = []
    for poly in edge_valid_faces:
        if len(poly) < 3:
            continue
        # (1) No vertex strictly inside
        has_vertex_inside = False
        for v in all_vertices_set:
            if on_boundary(poly, v):
                continue
            if Path(np.asarray(poly)).contains_point(v, radius=-1e-6):
                has_vertex_inside = True
                break
        if has_vertex_inside:
            continue
        # (2) No other face entirely inside this one
        has_face_inside = False
        for other in edge_valid_faces:
            if face_contains_other_face(poly, other):
                has_face_inside = True
                break
        if has_face_inside:
            continue
        room_regions.append(poly)
    return room_regions


# -----------------------------------------------------------------------------
# Step 4: Assign walls to rooms by centroid-to-boundary distance
# -----------------------------------------------------------------------------
def _assign_walls_to_rooms(wall_segments_dict, room_regions):
    """For each room polygon, collect wall geom names whose centroid is close to any edge."""
    wall_centroids = {}
    for geom_name, lines in wall_segments_dict.items():
        pts = []
        for (p, q) in lines:
            a, b = _canon(p), _canon(q)
            if _dist2(a, b) > 1e-12:
                pts.append(a)
                pts.append(b)
        if pts:
            cx = sum(x for x, _ in pts) / len(pts)
            cy = sum(y for _, y in pts) / len(pts)
            wall_centroids[geom_name] = (cx, cy)

    room_wall_geoms = []
    for poly in room_regions:
        wall_geoms_this_room = []
        n = len(poly)
        for geom_name, centroid in wall_centroids.items():
            min_dist_sq = float("inf")
            for j in range(n):
                u, v = poly[j], poly[(j + 1) % n]
                d_sq = _point_to_segment_dist_sq(centroid, u, v)
                min_dist_sq = min(min_dist_sq, d_sq)
            if min_dist_sq <= WALL_ROOM_ASSIGN_DIST_SQ:
                wall_geoms_this_room.append(geom_name)
        room_wall_geoms.append(wall_geoms_this_room)
    return room_wall_geoms


# -----------------------------------------------------------------------------
# Step 5: Assign floors and ceilings to rooms by interior point test
# -----------------------------------------------------------------------------
def _interior_sample_points(min_x, min_y, max_x, max_y):
    """Sample points inside the rectangle, excluding corners. Center + 4 points at 0.25/0.75."""
    w = max_x - min_x
    h = max_y - min_y
    if w <= 0 or h <= 0:
        return [(min_x + w / 2, min_y + h / 2)]
    pts = [(min_x + 0.5 * w, min_y + 0.5 * h)]
    for fx in (0.25, 0.75):
        for fy in (0.25, 0.75):
            pts.append((min_x + fx * w, min_y + fy * h))
    return pts


def _assign_floors_ceilings_to_rooms(floor_segments_dict, ceiling_segments_dict, room_regions):
    """Assign each floor/ceiling to a room if any interior sample point lies inside a room polygon."""
    room_floors = [set() for _ in range(len(room_regions))]
    room_ceilings = [set() for _ in range(len(room_regions))]

    for geom_name, rect in floor_segments_dict.items():
        min_x, min_y, max_x, max_y = rect
        sample_pts = _interior_sample_points(min_x, min_y, max_x, max_y)
        assigned = False
        for pt in sample_pts:
            if assigned:
                break
            for i, poly in enumerate(room_regions):
                if Path(np.asarray(poly)).contains_point(pt, radius=-1e-6):
                    room_floors[i].add(geom_name)
                    assigned = True
                    break

    for geom_name, rect in ceiling_segments_dict.items():
        min_x, min_y, max_x, max_y = rect
        sample_pts = _interior_sample_points(min_x, min_y, max_x, max_y)
        assigned = False
        for pt in sample_pts:
            if assigned:
                break
            for i, poly in enumerate(room_regions):
                if Path(np.asarray(poly)).contains_point(pt, radius=-1e-6):
                    room_ceilings[i].add(geom_name)
                    assigned = True
                    break

    return room_floors, room_ceilings


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def find_room_regions(wall_segments_dict, floor_segments_dict, ceiling_segments_dict):
    """
    1. Deduplicate walls (merge overlapping collinear segments) for graph construction.
    2. Find all vertices: segment endpoints + pairwise segment intersections.
    3. Build planar graph and find room regions as minimal faces.
    4. Assign each wall to a room by distance: if the wall centroid is close to any edge of the room polygon, assign that wall to the room.
    5. Assign each floor/ceiling segment to a room by testing multiple interior points (center + 0.25/0.75 points; corners excluded as ambiguous); if any point lies inside a room polygon, assign to that room.
    Returns (room_regions, vertices, room_geoms) where room_geoms[room_idx] = {"walls": [...], "floors": [...], "ceilings": [...]}.
    """
    raw = _flatten_wall_segments(wall_segments_dict)
    segments, segment_geom_names, segment_geom_with_segments = _deduplicate_wall_segments(raw)

    vertices = _collect_vertices(segments)
    edges, _, _ = _build_edges_with_geoms(
        segments, segment_geom_names, segment_geom_with_segments, vertices
    )

    room_regions = _find_minimal_faces(edges, vertices)
    room_wall_geoms = _assign_walls_to_rooms(wall_segments_dict, room_regions)
    room_floors, room_ceilings = _assign_floors_ceilings_to_rooms(
        floor_segments_dict, ceiling_segments_dict, room_regions
    )

    # Sort room indices by smallest vertex (min x, min y) so room with smallest coords gets idx=0
    def room_sort_key(i):
        poly = room_regions[i]
        min_x = min(v[0] for v in poly)
        min_y = min(v[1] for v in poly)
        return (min_x, min_y)

    order = sorted(range(len(room_regions)), key=room_sort_key)
    room_regions = [room_regions[i] for i in order]
    room_wall_geoms = [room_wall_geoms[i] for i in order]
    room_floors = [room_floors[i] for i in order]
    room_ceilings = [room_ceilings[i] for i in order]

    room_geoms = {
        i: {
            "walls": list(room_wall_geoms[i]),
            "floors": sorted(room_floors[i]),
            "ceilings": sorted(room_ceilings[i]),
        }
        for i in range(len(room_regions))
    }
    return room_regions, vertices, room_geoms
