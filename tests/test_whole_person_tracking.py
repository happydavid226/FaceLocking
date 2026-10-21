import math
import unittest

import numpy as np

from src.landmarks import Face
from src.whole_person_tracking import (
    BodyContinuityTracker,
    FaceBodyAssociator,
    IdentityState,
    PersonDetection,
    WholePersonTracker,
    add_face_guided_fallbacks,
    box_iou,
    evaluate_scenarios,
    face_guided_person,
)


def person(x, y=60, width=100, height=260):
    return PersonDetection(np.array([x, y, x + width, y + height], dtype=np.float32))


def face_for(body, x_offset=30):
    x1, y1, _, _ = body.box
    box = np.array([x1 + x_offset, y1 + 20, 40, 50], dtype=np.float32)
    points = np.array([[x1 + 38, y1 + 34], [x1 + 62, y1 + 34], [x1 + 50, y1 + 44],
                       [x1 + 40, y1 + 57], [x1 + 60, y1 + 57]], dtype=np.float32)
    return Face(box, points, 0.98)


class WholePersonTrackingTests(unittest.TestCase):
    def test_face_body_association_requires_containment(self):
        body = person(100)
        associator = FaceBodyAssociator()
        self.assertGreater(associator.score(face_for(body), body), 0)
        outside = Face(np.array([500, 20, 40, 50], dtype=np.float32),
                       np.zeros((5, 2), dtype=np.float32), 0.9)
        self.assertEqual(associator.score(outside, body), -1.0)
        self.assertAlmostEqual(box_iou(body.box, body.box), 1.0)

    def test_confidence_decays_exponentially_while_body_continues(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (True, 0.8), decay_rate=0.5,
                                     temporary_floor=0.01, max_temporary_frames=5)
        first = tracker.update([face_for(target)], [target], (480, 640, 3))
        missing = tracker.update([], [target], (480, 640, 3))
        self.assertEqual(first.state, IdentityState.CONFIRMED)
        self.assertEqual(missing.state, IdentityState.TEMPORARY)
        self.assertAlmostEqual(missing.confidence, first.confidence * math.exp(-0.5), places=6)

    def test_face_loss_marks_body_temporary_and_matching_return_reconfirms(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (True, 0.9), decay_rate=0.1,
                                     temporary_floor=0.01, max_temporary_frames=5)
        confirmed = tracker.update([face_for(target)], [target], (480, 640, 3))
        missing = tracker.update([], [target], (480, 640, 3))
        returned = tracker.update([face_for(target)], [target], (480, 640, 3))
        self.assertEqual(confirmed.state, IdentityState.CONFIRMED)
        self.assertEqual(missing.state, IdentityState.TEMPORARY)
        self.assertIs(missing.person, target)
        self.assertEqual(returned.state, IdentityState.CONFIRMED)
        self.assertIs(returned.person, target)
        self.assertIsNotNone(returned.face)

    def test_face_and_body_detector_dropout_starts_temporary_from_last_body_box(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (face.confidence > 0.95, 0.9), decay_rate=0.1,
                                     temporary_floor=0.01, max_temporary_frames=3)
        tracker.update([face_for(target)], [target], (480, 640, 3))
        missing = tracker.update([], [], (480, 640, 3))
        self.assertEqual(missing.state, IdentityState.TEMPORARY)
        self.assertIs(missing.person, target)
        self.assertEqual(missing.reason, "Face lost; holding last body box briefly")

    def test_unassociated_face_uses_grace_then_becomes_temporary(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (face.confidence > 0.95, 0.9), decay_rate=0.1,
                                     temporary_floor=0.01, max_temporary_frames=3)
        tracker.update([face_for(target)], [target], (480, 640, 3))
        unrelated_face = Face(np.array([500, 30, 40, 50], dtype=np.float32),
                              np.zeros((5, 2), dtype=np.float32), 0.9)
        missing = tracker.update([unrelated_face], [], (480, 640, 3))
        self.assertEqual(missing.state, IdentityState.CONFIRMED)
        self.assertEqual(missing.reason, "Face detected; smoothing identity mismatch")
        missing = tracker.update([unrelated_face], [], (480, 640, 3))
        self.assertEqual(missing.state, IdentityState.CONFIRMED)
        missing = tracker.update([unrelated_face], [], (480, 640, 3))
        self.assertEqual(missing.state, IdentityState.CONFIRMED)
        missing = tracker.update([unrelated_face], [], (480, 640, 3))
        self.assertEqual(missing.state, IdentityState.TEMPORARY)
        self.assertIs(missing.person, target)

    def test_unverified_face_on_locked_body_stays_temporary(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (face.confidence > 0.95, 0.9), max_temporary_frames=5)
        self.assertEqual(tracker.update([face_for(target)], [target], (480, 640, 3)).state,
                         IdentityState.CONFIRMED)
        distractor = face_for(target)
        distractor = Face(distractor.box, distractor.points, 0.5)
        result = tracker.update([distractor], [target], (480, 640, 3))
        self.assertEqual(result.state, IdentityState.CONFIRMED)
        result = tracker.update([distractor], [target], (480, 640, 3))
        self.assertEqual(result.state, IdentityState.CONFIRMED)
        result = tracker.update([distractor], [target], (480, 640, 3))
        self.assertEqual(result.state, IdentityState.CONFIRMED)
        result = tracker.update([distractor], [target], (480, 640, 3))
        self.assertEqual(result.state, IdentityState.TEMPORARY)
        self.assertLess(result.confidence, 1.0)
        self.assertIs(result.person, target)
        self.assertIs(result.face, distractor)
        self.assertEqual(result.reason, "Face unclear; locked body retained temporarily")

    def test_unverified_face_becomes_unknown_after_lock_timeout(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (face.confidence > 0.95, 0.9),
                                     max_temporary_frames=1, temporary_floor=0.01,
                                     face_mismatch_grace=0)
        tracker.update([face_for(target)], [target], (480, 640, 3))
        uncertain = face_for(target)
        uncertain = Face(uncertain.box, uncertain.points, 0.5)
        self.assertEqual(tracker.update([uncertain], [target], (480, 640, 3)).state,
                         IdentityState.TEMPORARY)
        expired = tracker.update([uncertain], [target], (480, 640, 3))
        self.assertEqual(expired.state, IdentityState.UNKNOWN)

    def test_matching_visible_face_recovers_even_when_body_box_jumps(self):
        target = person(100)
        moved = person(360)
        tracker = WholePersonTracker(lambda face: (True, 0.9), association_threshold=0.05)
        self.assertEqual(tracker.update([face_for(target)], [target], (480, 640, 3)).state,
                         IdentityState.CONFIRMED)
        recovered = tracker.update([face_for(moved)], [moved], (480, 640, 3))
        self.assertEqual(recovered.state, IdentityState.CONFIRMED)
        self.assertIs(recovered.person, moved)

    def test_expired_identity_still_renders_visible_body_as_unknown(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (True, 0.9), decay_rate=1.0,
                                     temporary_floor=0.5, max_temporary_frames=4)
        tracker.update([face_for(target)], [target], (480, 640, 3))
        result = tracker.update([], [target], (480, 640, 3))
        self.assertEqual(result.state, IdentityState.UNKNOWN)
        self.assertIs(result.person, target)
        self.assertIsNotNone(result.signal)

    def test_body_can_start_unknown_then_be_confirmed_by_a_later_face(self):
        target = person(200)
        tracker = WholePersonTracker(lambda face: (True, 0.9))
        body_only = tracker.update([], [target], (480, 640, 3))
        self.assertEqual(body_only.state, IdentityState.UNKNOWN)
        self.assertIs(body_only.person, target)
        confirmed = tracker.update([face_for(target)], [target], (480, 640, 3))
        self.assertEqual(confirmed.state, IdentityState.CONFIRMED)

    def test_scenarios_pass_and_centre_signal_is_directional(self):
        report = evaluate_scenarios()
        self.assertTrue(all(item["passed"] for item in report.values()))
        self.assertEqual(report["head_turn"]["states"][-1], "Confirmed")
        target = person(20)
        tracker = WholePersonTracker(lambda face: (True, 0.9))
        result = tracker.update([face_for(target)], [target], (480, 640, 3))
        self.assertEqual(result.signal.horizontal, "LEFT")

    def test_face_guided_fallback_covers_closeup_when_hog_has_no_box(self):
        target = person(250)
        face = face_for(target)
        fallback = face_guided_person(face, (480, 640, 3))
        self.assertEqual(len(add_face_guided_fallbacks([face], [], (480, 640, 3))), 1)
        self.assertGreater(FaceBodyAssociator().score(face, fallback), 0)

    def test_body_continuity_tracker_predicts_after_detector_dropout(self):
        import cv2
        first = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.rectangle(first, (40, 40), (90, 130), (255, 255, 255), -1)
        second = np.zeros_like(first)
        cv2.rectangle(second, (45, 40), (95, 130), (255, 255, 255), -1)
        tracker = BodyContinuityTracker(max_missing=3)
        self.assertTrue(tracker.start(first, person(40, 40, 50, 90)))
        prediction = tracker.predict(second)
        self.assertIsNotNone(prediction)
        self.assertGreater(float(prediction.box[0]), 40)

    def test_body_continuity_holds_last_box_during_short_tracker_failure(self):
        class FailingTracker:
            def update(self, frame):
                return False, None

        tracker = BodyContinuityTracker(max_missing=3)
        tracker._tracker = FailingTracker()
        tracker.last_box = np.array([20, 20, 80, 120], dtype=np.float32)
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        self.assertIsNotNone(tracker.predict(frame))
        self.assertIsNotNone(tracker.predict(frame))
        self.assertIsNone(tracker.predict(frame))


if __name__ == "__main__":
    unittest.main()
