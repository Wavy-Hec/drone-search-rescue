"""Instance-id segmentation -> objects / classes / boxes, and the top-down map (numpy + matplotlib only).

Run: ``python tests/test_sar_labels.py`` (or with pytest).
"""

from __future__ import annotations

import numpy as np
from _runner import main_for


def _frame():
    seg = np.ones((40, 60), dtype=np.int32)  # UNLABELLED ground
    seg[0:2, 0:60] = 0  # BACKGROUND (sky)
    seg[5:15, 10:20] = 3  # person_02 body mesh
    seg[8:12, 20:30] = 4  # person_02 arm mesh (same object, other mesh)
    seg[20:30, 40:55] = 7  # a crate
    seg[30:35, 0:5] = 9  # unlabeled tree (filtered out of the semantics)
    info = {
        "idToLabels": {
            "0": "BACKGROUND",
            "1": "UNLABELLED",
            "3": "/World/sar/objects/person_02/Root/a/body",
            "4": "/World/sar/objects/person_02/Root/a/arms",
            "7": "/World/sar/objects/crate_05/Root/mesh",
            "9": "/World/sar/objects/tree_01/Root/leaves",
        },
        "idToSemantics": {
            "0": {"class": "BACKGROUND"},
            "1": {"class": "UNLABELLED"},
            "3": {"class": "person"},
            "4": {"class": "person"},
            "7": {"class": "crate"},
        },
    }
    return seg, info


def test_instance_boxes_group_meshes_by_object():
    from drone_lab.sar.labels import instance_boxes, object_name_from_path

    seg, info = _frame()
    boxes = {b.name: b for b in instance_boxes(seg[..., None], info)}
    assert set(boxes) == {"person_02", "crate_05"}
    p = boxes["person_02"]
    assert p.label == "person" and p.class_id == 0 and set(p.ids) == {3, 4}
    assert p.bbox_xyxy == (10, 5, 29, 14) and p.pixels == 100 + 40
    c = boxes["crate_05"]
    assert c.label == "crate" and c.class_id == 2 and c.bbox_xyxy == (40, 20, 54, 29)
    assert object_name_from_path("/World/sar/objects/barrel_01/x") == "barrel_01"
    assert object_name_from_path("/World/envs/env_0/Robot/base_link") is None


def test_semantic_label_forms_and_class_map():
    from drone_lab.sar.labels import class_map, colorize_classes, semantic_label

    assert semantic_label({"class": "person"}) == "person"
    assert semantic_label("class: cone") == "cone"
    assert semantic_label({"class": "person,forklift"}) == "person,forklift"  # double labels are visible, not hidden
    assert semantic_label(None) is None and semantic_label({}) is None
    assert semantic_label({"class": "BACKGROUND"}) is None and semantic_label({"class": "UNLABELLED"}) is None
    seg, info = _frame()
    cm = class_map(seg, info)
    assert cm[6, 12] == 0 and cm[25, 45] == 2 and cm[32, 2] == -1 and cm[0, 0] == -1
    rgb = colorize_classes(cm)
    assert rgb.shape == (40, 60, 3) and tuple(rgb[25, 45]) == (255, 225, 25)


if __name__ == "__main__":
    main_for(globals(), "test_sar_labels")
