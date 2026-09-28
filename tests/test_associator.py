"""
Unit tests for AnatomicalAssociator: PPE mapping to persons.
"""

import pytest
from core.associator import AnatomicalAssociator
from core.detector import (
    Detection,
    CLASS_PERSON,
    CLASS_FACE_MASK,
    CLASS_MASK_INCORRECT,
    CLASS_GOWN,
    CLASS_GLOVES,
    CLASS_SHOE_COVERS
)


def test_associator_maps_mask_to_person_head():
    associator = AnatomicalAssociator()

    # Person 1 at x: 100-200, y: 100-300 (head is ~100-140)
    person = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    # Mask inside person's head region
    mask = Detection(bbox=(130, 115, 170, 135), class_name=CLASS_FACE_MASK)
    # Gown on torso (140-250)
    gown = Detection(bbox=(110, 150, 190, 240), class_name=CLASS_GOWN)
    # Gloves near hands
    gloves = Detection(bbox=(95, 200, 125, 230), class_name=CLASS_GLOVES)

    profiles = associator.associate([person, mask, gown, gloves])

    assert len(profiles) == 1
    prof = profiles[0]
    assert prof.mask_status == CLASS_FACE_MASK
    assert prof.has_gown is True
    assert prof.has_gloves is True
    assert prof.has_shoe_covers is False


def test_associator_multi_person_separation():
    associator = AnatomicalAssociator()

    # Person A (x: 100-200) with Mask
    person_a = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    mask_a = Detection(bbox=(130, 115, 170, 135), class_name=CLASS_FACE_MASK)

    # Person B (x: 500-600) with Incorrect Mask and Gown
    person_b = Detection(bbox=(500, 100, 600, 300), class_name=CLASS_PERSON)
    mask_b = Detection(bbox=(530, 130, 570, 145), class_name=CLASS_MASK_INCORRECT)
    gown_b = Detection(bbox=(510, 150, 590, 240), class_name=CLASS_GOWN)

    profiles = associator.associate([person_a, person_b, mask_a, mask_b, gown_b])

    assert len(profiles) == 2
    # Find matching profile for A
    prof_a = next(p for p in profiles if p.person_det.bbox == person_a.bbox)
    prof_b = next(p for p in profiles if p.person_det.bbox == person_b.bbox)

    assert prof_a.mask_status == CLASS_FACE_MASK
    assert prof_a.has_gown is False

    assert prof_b.mask_status == CLASS_MASK_INCORRECT
    assert prof_b.has_gown is True

