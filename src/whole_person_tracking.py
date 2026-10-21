"""CPU-only whole-person tracking with honest face continuity states.

The module deliberately separates *where a person is* from *who they are*.
An observed face can confirm an identity, while a body track can only carry
that identity through a short facial occlusion.  Once the body continuity or
the exponentially decayed confidence is exhausted, the result is UNKNOWN;
the tracker never silently promotes a different person to the old identity.

Run a deterministic evaluation without a camera with::

    python -m src.whole_person_tracking --evaluate

Run the webcam demo after enrolling a target with::

    python -m src.whole_person_tracking --target "Isaac" --camera 0
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from .align import align_face
from .landmarks import Face


class IdentityState(str, Enum):
    """The only identity claims exposed by the tracker."""

    CONFIRMED = "Confirmed"
    TEMPORARY = "Temporary"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class PersonDetection:
    """A person bounding box in ``(x1, y1, x2, y2)`` image coordinates."""

    box: np.ndarray
    confidence: float = 1.0


@dataclass(frozen=True)
class Association:
    """The face/person pairing selected for one frame."""

    person: PersonDetection
    face: Optional[Face]
    score: float


@dataclass(frozen=True)
class CentreSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


@dataclass(frozen=True)
class WholePersonResult:
    state: IdentityState
    confidence: float
    person: Optional[PersonDetection]
    face: Optional[Face]
    signal: Optional[CentreSignal]
    reason: str
    frame_index: int
    match_score: Optional[float] = None


def xyxy_from_xywh(box: Sequence[float]) -> np.ndarray:
    x, y, width, height = map(float, box[:4])
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    area_b = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def box_center(box: Sequence[float]) -> np.ndarray:
    return (np.asarray(box[:2], dtype=np.float32) + np.asarray(box[2:4], dtype=np.float32)) / 2.0


class HOGPersonDetector:
    """OpenCV's built-in CPU HOG people detector.

    It is intentionally conservative and is a useful baseline for a webcam
    demo.  A detector object with the same ``detect(frame)`` method can be
    injected into :class:`WholePersonTracker` for a stronger model or tests.
    """

    def __init__(self, hit_threshold: float = 0.0, scale: float = 1.05,
                 win_stride: tuple[int, int] = (8, 8)):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = float(hit_threshold)
        self.scale = float(scale)
        self.win_stride = win_stride

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        boxes, weights = self.hog.detectMultiScale(
            frame, self.hit_threshold, self.win_stride, (8, 8), self.scale)
        results = []
        for box, weight in zip(boxes, weights):
            results.append(PersonDetection(xyxy_from_xywh(box), float(weight)))
        return results


class PosePersonDetector:
    """CPU MediaPipe pose detector for body-only frames.

    Pose landmarks are independent of face detection, so a turned head or a
    face outside the image can still produce a body box.  MediaPipe is an
    optional runtime here; when unavailable this detector simply returns no
    boxes and HOG/CSRT continue to provide the baseline path.
    """

    def __init__(self, min_detection_confidence: float = 0.45,
                 min_tracking_confidence: float = 0.45):
        self.pose = None
        try:
            import mediapipe as mp
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
        except (ImportError, AttributeError, RuntimeError):
            self.pose = None

    @property
    def available(self) -> bool:
        return self.pose is not None

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        if self.pose is None:
            return []
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return []
        visible = [landmark for landmark in result.pose_landmarks.landmark
                   if landmark.visibility >= 0.35 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1]
        if len(visible) < 4:
            return []
        x_values = [landmark.x * width for landmark in visible]
        y_values = [landmark.y * height for landmark in visible]
        x1, x2 = max(0.0, min(x_values)), min(float(width), max(x_values))
        y1, y2 = max(0.0, min(y_values)), min(float(height), max(y_values))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return []
        confidence = float(np.mean([landmark.visibility for landmark in visible]))
        return [PersonDetection(np.array([x1, y1, x2, y2], dtype=np.float32), confidence)]

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()


def merge_person_detections(*groups: Iterable[PersonDetection], iou_threshold: float = 0.35
                            ) -> list[PersonDetection]:
    """Combine HOG/pose boxes without emitting duplicate bodies."""
    merged: list[PersonDetection] = []
    for group in groups:
        for candidate in group:
            overlap = next((index for index, existing in enumerate(merged)
                            if box_iou(candidate.box, existing.box) >= iou_threshold), None)
            if overlap is None:
                merged.append(candidate)
            elif candidate.confidence > merged[overlap].confidence:
                merged[overlap] = candidate
    return merged


class BodyContinuityTracker:
    """Carry a confirmed body through detector dropouts with CSRT.

    HOG is an object detector, not a temporal tracker: it can stop producing a
    box when only part of a person remains in frame.  This small OpenCV CPU
    tracker is seeded only from an observed body and is used as a measurement
    during a facial-loss gap.  It is reset when identity becomes UNKNOWN, so
    its pixels never create a new identity claim.
    """

    def __init__(self, max_missing: int = 24):
        if not isinstance(max_missing, int) or max_missing < 1:
            raise ValueError("max_missing must be a positive integer")
        self.max_missing = max_missing
        self._tracker = None
        self.last_box: Optional[np.ndarray] = None
        self.missing = 0

    @staticmethod
    def _factory():
        factory = getattr(cv2, "TrackerCSRT_create", None)
        if factory is not None:
            return factory
        legacy = getattr(cv2, "legacy", None)
        return getattr(legacy, "TrackerCSRT_create", None) if legacy is not None else None

    @staticmethod
    def _clip(box: Sequence[float], shape: Sequence[int]) -> Optional[np.ndarray]:
        height, width = shape[:2]
        x1, y1, x2, y2 = map(float, box[:4])
        x1, y1 = max(0.0, min(x1, width - 2.0)), max(0.0, min(y1, height - 2.0))
        x2, y2 = max(x1 + 2.0, min(x2, width)), max(y1 + 2.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def reset(self) -> None:
        self._tracker = None
        self.last_box = None
        self.missing = 0

    def start(self, frame: np.ndarray, person: PersonDetection) -> bool:
        factory = self._factory()
        box = self._clip(person.box, frame.shape)
        if factory is None or box is None:
            self.reset()
            return False
        x1, y1, x2, y2 = box
        tracker = factory()
        initialized = tracker.init(frame, (int(round(float(x1))), int(round(float(y1))),
                                           int(round(float(x2 - x1))), int(round(float(y2 - y1)))))
        self._tracker = tracker
        self.last_box = box
        self.missing = 0
        return initialized is not False

    def predict(self, frame: np.ndarray) -> Optional[PersonDetection]:
        if self._tracker is None:
            return None
        ok, raw = self._tracker.update(frame)
        if not ok:
            self.missing += 1
            if self.missing >= self.max_missing:
                self.reset()
                return None
            # Keep the last trustworthy body location during a short tracker
            # hiccup. WholePersonTracker will decay confidence while this
            # conservative hold prevents a stationary body from disappearing.
            return PersonDetection(self.last_box.copy(), confidence=0.05) if self.last_box is not None else None
        x, y, width, height = raw
        box = self._clip((x, y, x + width, y + height), frame.shape)
        if box is None:
            self.missing += 1
            if self.missing >= self.max_missing:
                self.reset()
                return None
            return PersonDetection(self.last_box.copy(), confidence=0.05) if self.last_box is not None else None
        self.last_box, self.missing = box, 0
        return PersonDetection(box, confidence=0.18)


class FaceBodyAssociator:
    """Map a face to the most plausible body using containment and distance."""

    def __init__(self, min_face_fraction: float = 0.02, max_face_fraction: float = 0.65):
        self.min_face_fraction = float(min_face_fraction)
        self.max_face_fraction = float(max_face_fraction)

    @staticmethod
    def face_box(face: Face) -> np.ndarray:
        return xyxy_from_xywh(face.box)

    def score(self, face: Face, person: PersonDetection) -> float:
        fbox = self.face_box(face)
        pbox = person.box
        face_width = max(1.0, fbox[2] - fbox[0])
        body_width = max(1.0, pbox[2] - pbox[0])
        fraction = float(face_width / body_width)
        if fraction < self.min_face_fraction or fraction > self.max_face_fraction:
            return -1.0
        face_centre = box_center(fbox)
        inside_x = pbox[0] <= face_centre[0] <= pbox[2]
        inside_y = pbox[1] <= face_centre[1] <= pbox[3]
        if not (inside_x and inside_y):
            return -1.0
        distance = np.linalg.norm(face_centre - box_center(pbox))
        diagonal = max(float(np.linalg.norm(pbox[2:] - pbox[:2])), 1.0)
        containment = box_iou(fbox, pbox)
        return float(0.60 * containment + 0.40 * (1.0 - min(distance / diagonal, 1.0)))

    def associate(self, faces: Iterable[Face], persons: Iterable[PersonDetection]) -> list[Association]:
        available = list(persons)
        pairs: list[Association] = []
        for face in faces:
            ranked = [(self.score(face, person), person) for person in available]
            if not ranked:
                continue
            score, person = max(ranked, key=lambda item: item[0])
            if score >= 0:
                pairs.append(Association(person, face, score))
        # A body can only own one face in a frame; retain the strongest pairing.
        unique: dict[int, Association] = {}
        for pair in pairs:
            key = id(pair.person)
            if key not in unique or pair.score > unique[key].score:
                unique[key] = pair
        return list(unique.values())


class WholePersonTracker:
    """Lock a confirmed face identity to its associated body over time.

    ``identity_match`` receives a face and returns ``(is_target, score)``.
    Scores are expected to be cosine-like values in ``[-1, 1]``.  A confirmed
    observation seeds confidence from the score; missing faces multiply the
    confidence by ``exp(-decay_rate * dt)``.  Body continuity is evidence of
    location only and cannot turn an UNKNOWN person into CONFIRMED.
    """

    def __init__(self, identity_match: Callable[[Face], tuple[bool, Optional[float]]],
                 *, decay_rate: float = 0.35, temporary_floor: float = 0.20,
                 max_temporary_frames: int = 24, association_threshold: float = 0.05,
                 ema_alpha: float = 0.35, dead_zone: float = 0.08,
                 face_mismatch_grace: int = 3):
        if decay_rate <= 0 or not 0 < temporary_floor < 1:
            raise ValueError("decay_rate must be positive and temporary_floor in (0,1)")
        if not isinstance(max_temporary_frames, int) or max_temporary_frames < 0:
            raise ValueError("max_temporary_frames must be a non-negative integer")
        if not isinstance(face_mismatch_grace, int) or face_mismatch_grace < 0:
            raise ValueError("face_mismatch_grace must be a non-negative integer")
        if not 0 < ema_alpha <= 1 or not 0 <= dead_zone < 1:
            raise ValueError("ema_alpha must be in (0,1] and dead_zone in [0,1)")
        self.identity_match = identity_match
        self.decay_rate = float(decay_rate)
        self.temporary_floor = float(temporary_floor)
        self.max_temporary_frames = max_temporary_frames
        self.face_mismatch_grace = face_mismatch_grace
        self.association_threshold = float(association_threshold)
        self.ema_alpha, self.dead_zone = float(ema_alpha), float(dead_zone)
        self.associator = FaceBodyAssociator()
        self.state = IdentityState.UNKNOWN
        self.confidence = 0.0
        self.last_person: Optional[PersonDetection] = None
        self.last_face: Optional[Face] = None
        self.smooth_center: Optional[np.ndarray] = None
        self.temporary_frames = 0
        self.unverified_face_frames = 0
        self.frame_index = 0

    @staticmethod
    def _confidence(score: float) -> float:
        return float(np.clip((float(score) + 1.0) / 2.0, 0.0, 1.0))

    def _signal(self, shape: Sequence[int]) -> Optional[CentreSignal]:
        if self.smooth_center is None:
            return None
        height, width = shape[:2]
        error_x = float((self.smooth_center[0] - width / 2) / max(width / 2, 1))
        error_y = float((self.smooth_center[1] - height / 2) / max(height / 2, 1))
        horizontal = "LEFT" if error_x < -self.dead_zone else "RIGHT" if error_x > self.dead_zone else "CENTER"
        vertical = "UP" if error_y < -self.dead_zone else "DOWN" if error_y > self.dead_zone else "CENTER"
        return CentreSignal(error_x, error_y, horizontal, vertical)

    def _body_match(self, persons: Sequence[PersonDetection]) -> Optional[PersonDetection]:
        if self.last_person is None or not persons:
            return None
        previous = box_center(self.last_person.box)
        diagonal = max(float(np.linalg.norm(self.last_person.box[2:] - self.last_person.box[:2])), 1.0)
        ranked = []
        for person in persons:
            displacement = np.linalg.norm(box_center(person.box) - previous) / diagonal
            score = box_iou(self.last_person.box, person.box) - 0.30 * displacement
            ranked.append((score, person))
        score, person = max(ranked, key=lambda item: item[0])
        return person if score >= self.association_threshold else None

    def _remember(self, person: PersonDetection, shape: Sequence[int]) -> CentreSignal:
        self.last_person = person
        raw = box_center(person.box)
        self.smooth_center = raw if self.smooth_center is None else (
            self.ema_alpha * raw + (1.0 - self.ema_alpha) * self.smooth_center)
        return self._signal(shape)

    def update(self, faces: Iterable[Face], persons: Iterable[PersonDetection],
               frame_shape: Sequence[int]) -> WholePersonResult:
        self.frame_index += 1
        faces, persons = list(faces), list(persons)
        associations = self.associator.associate(faces, persons)
        body = self._body_match(persons)
        held_body = False
        # A detector can miss the body while unrelated faces remain visible.
        # The state is about the previously locked body: if no current body
        # measurement matches it, hold that body briefly unless a positive
        # face match below reacquires the target elsewhere.
        if (body is None and self.last_person is not None
                and self.state in (IdentityState.CONFIRMED, IdentityState.TEMPORARY)
                and self.temporary_frames < self.max_temporary_frames):
            body = self.last_person
            held_body = True
        chosen: Optional[Association] = None
        chosen_identity: Optional[tuple[bool, Optional[float]]] = None
        uncertain_face: Optional[Face] = None
        body_pair: Optional[Association] = None
        if body is not None:
            for pair in associations:
                if pair.person is body:
                    body_pair = pair
                    break

        # Face identity is stronger evidence than the body detector's box
        # overlap. A shifted HOG/pose box must not turn a visible target face
        # into Unknown just because geometric association missed for one frame.
        # Search for a positive identity match first, including during recovery
        # from Temporary and while body tracking catches up. Keep unassociated
        # face detections in the check too: a slightly wrong body box should
        # not make an otherwise visible, matching face look absent.
        matches: list[tuple[float, Association, tuple[bool, Optional[float]]]] = []
        uncertain_pairs: list[tuple[Association, tuple[bool, Optional[float]]]] = []
        pair_by_face = {id(pair.face): pair for pair in associations}
        for face in faces:
            pair = pair_by_face.get(id(face))
            if pair is None:
                if body is None:
                    continue
                pair = Association(body, face, 0.0)
            identity = self.identity_match(pair.face)
            is_target, score = identity
            if is_target and score is not None:
                matches.append((float(score), pair, identity))
            else:
                uncertain_pairs.append((pair, identity))
        if matches:
            _, chosen, chosen_identity = max(matches, key=lambda item: item[0])
        elif body_pair is not None:
            # Preserve the locked body's face observation for the uncertainty
            # path below; a failed match alone is not enough to discard lock.
            chosen = body_pair
            chosen_identity = next((identity for pair, identity in uncertain_pairs
                                    if pair.face is body_pair.face), None)
        elif uncertain_pairs and body is not None:
            chosen, chosen_identity = max(uncertain_pairs, key=lambda item: item[0].score)

        if chosen is not None:
            is_target, score = (chosen_identity if chosen_identity is not None
                                else self.identity_match(chosen.face))
            if is_target and score is not None:
                self.state = IdentityState.CONFIRMED
                self.confidence = max(self.confidence * 0.65, self._confidence(score))
                self.temporary_frames = 0
                self.unverified_face_frames = 0
                self.last_face = chosen.face
                signal = self._remember(chosen.person, frame_shape)
                return WholePersonResult(self.state, self.confidence, chosen.person, chosen.face,
                                         signal, "Face identity confirmed", self.frame_index, score)
            if (body is not None and self.state in
                    (IdentityState.CONFIRMED, IdentityState.TEMPORARY)):
                # A face on the already-locked body can be turned, blurred,
                # partially occluded, or briefly misclassified. Keep the body
                # lock as Temporary; only a positive target match reconfirms.
                uncertain_face = chosen.face
                self.unverified_face_frames += 1
                if (self.state == IdentityState.CONFIRMED
                        and self.unverified_face_frames <= self.face_mismatch_grace):
                    # A transient ArcFace false negative should not flicker
                    # the state when the face is visibly still present.
                    self.confidence *= math.exp(-self.decay_rate * 0.1)
                    signal = self._remember(body, frame_shape)
                    return WholePersonResult(
                        IdentityState.CONFIRMED, self.confidence, body, uncertain_face,
                        signal, "Face detected; smoothing identity mismatch", self.frame_index, score)
            else:
                # Before a face/body lock exists, a failed face match cannot
                # inherit somebody else's identity.
                self.state = IdentityState.UNKNOWN
                self.confidence = 0.0
                self.last_face = None
                signal = self._remember(chosen.person, frame_shape)
                return WholePersonResult(self.state, self.confidence, chosen.person, chosen.face, signal,
                                         "Face does not match target identity", self.frame_index, score)

        if body is not None and self.state in (IdentityState.CONFIRMED, IdentityState.TEMPORARY):
            self.last_face = None
            self.unverified_face_frames = 0
            self.temporary_frames += 1
            self.confidence *= math.exp(-self.decay_rate)
            if self.temporary_frames <= self.max_temporary_frames and self.confidence >= self.temporary_floor:
                self.state = IdentityState.TEMPORARY
                signal = self._remember(body, frame_shape)
                reason = ("Face unclear; locked body retained temporarily" if uncertain_face is not None else
                          "Face lost; holding last body box briefly" if held_body else
                          "Face temporarily unavailable; body continuity retained")
                return WholePersonResult(self.state, self.confidence, body, uncertain_face, signal, reason,
                                         self.frame_index)

        self.state = IdentityState.UNKNOWN
        # Confidence is an identity claim, so it must be cleared as soon as
        # continuity expires. Keeping a sub-floor value internally can leak
        # stale confidence into a later body-only result or re-confirmation.
        self.confidence = 0.0
        self.temporary_frames += 1
        self.last_face = None
        if body is not None:
            # The body can remain visible after identity confidence expires.
            # Keep rendering it as UNKNOWN; never turn this observation back
            # into a target identity without a new positive face match.
            signal = self._remember(body, frame_shape)
            return WholePersonResult(self.state, 0.0, body, None, signal,
                                     "Body visible; identity confidence expired", self.frame_index)
        if self.state == IdentityState.UNKNOWN and persons:
            # Body detection is allowed to start before identity is known.
            # This is essential when a person enters side-on or the first face
            # frame is too small; only a later positive face match can confirm
            # the target identity.
            bootstrap = max(persons, key=lambda item: item.confidence)
            signal = self._remember(bootstrap, frame_shape)
            return WholePersonResult(self.state, 0.0, bootstrap, None, signal,
                                     "Body detected; waiting for face confirmation", self.frame_index)
        return WholePersonResult(self.state, self.confidence, None, None, None,
                                 "No trustworthy face/body continuity", self.frame_index)


def draw_result(frame: np.ndarray, result: WholePersonResult,
                locked_name: Optional[str] = None) -> np.ndarray:
    """Draw the result after inference, keeping overlays out of face crops."""
    if result.person is not None:
        x1, y1, x2, y2 = np.rint(result.person.box).astype(int)
        color = {IdentityState.CONFIRMED: (0, 210, 0), IdentityState.TEMPORARY: (0, 180, 255),
                 IdentityState.UNKNOWN: (0, 80, 255)}[result.state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        identity = (locked_name or "Locked person") if result.state in (
            IdentityState.CONFIRMED, IdentityState.TEMPORARY) else "Unknown"
        label = f"{result.state.value}: {identity} {result.confidence:.2f}"
        cv2.putText(frame, label, (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if result.face is not None:
        fx, fy, fw, fh = np.rint(result.face.box).astype(int)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 190, 0), 1)
    if result.signal is not None:
        cv2.putText(frame, f"centre: {result.signal.horizontal}/{result.signal.vertical}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    identity = (locked_name or "Locked person") if result.state in (
        IdentityState.CONFIRMED, IdentityState.TEMPORARY) else "Unknown"
    cv2.putText(frame, f"{result.state.value}: {identity} | confidence {result.confidence:.3f}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    reason = result.reason[:58]
    cv2.putText(frame, reason, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (220, 220, 220), 1, cv2.LINE_AA)
    if result.match_score is not None:
        cv2.putText(frame, f"match score={result.match_score:.3f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def face_guided_person(face: Face, frame_shape: Sequence[int]) -> PersonDetection:
    """Build a conservative upper-body box when HOG misses a close-up subject."""
    height, width = frame_shape[:2]
    x, y, w, h = map(float, face.box)
    return PersonDetection(np.array([
        max(0.0, x - 1.0 * w), max(0.0, y - 0.6 * h),
        min(float(width), x + 2.0 * w), min(float(height), y + 6.0 * h),
    ], dtype=np.float32), confidence=0.25)


def add_face_guided_fallbacks(faces: Iterable[Face], persons: Iterable[PersonDetection],
                              frame_shape: Sequence[int]) -> list[PersonDetection]:
    """Keep HOG boxes, adding a body prior only for faces it did not cover."""
    faces, persons = list(faces), list(persons)
    covered = {id(pair.face) for pair in FaceBodyAssociator().associate(faces, persons)}
    return persons + [face_guided_person(face, frame_shape) for face in faces if id(face) not in covered]


def _scenario_person(x: float, y: float = 60, width: float = 100, height: float = 260) -> PersonDetection:
    return PersonDetection(np.array([x, y, x + width, y + height], dtype=np.float32))


def _scenario_face(person: PersonDetection, visible: bool = True, confidence: float = 0.98) -> list[Face]:
    if not visible:
        return []
    x1, y1, x2, _ = person.box
    box = np.array([x1 + 30, y1 + 20, 40, 50], dtype=np.float32)
    points = np.array([[x1 + 38, y1 + 34], [x1 + 62, y1 + 34], [x1 + 50, y1 + 44],
                       [x1 + 40, y1 + 57], [x1 + 60, y1 + 57]], dtype=np.float32)
    return [Face(box, points, confidence)]


def evaluate_scenarios() -> dict[str, dict[str, object]]:
    """Run head-turn, occlusion, and crossing checks without a camera/model."""
    def match(face: Face) -> tuple[bool, float]:
        return True, 0.92

    tracker = WholePersonTracker(match, decay_rate=0.30, temporary_floor=0.20, max_temporary_frames=5)
    person = _scenario_person(240)
    head_turn = [tracker.update(_scenario_face(person, visible=i < 2 or i == 5), [person], (480, 640, 3)).state.value
                 for i in range(6)]
    tracker = WholePersonTracker(match, decay_rate=0.50, temporary_floor=0.35, max_temporary_frames=3)
    occlusion_person = _scenario_person(220)
    occlusion = [tracker.update(_scenario_face(occlusion_person, visible=i == 0), [occlusion_person], (480, 640, 3)).state.value
                 for i in range(5)]
    def crossing_match(face: Face) -> tuple[bool, float]:
        # The lower-confidence face is the crossing distractor.
        return face.confidence > 0.9, 0.92 if face.confidence > 0.9 else 0.2

    tracker = WholePersonTracker(crossing_match, decay_rate=0.35, max_temporary_frames=4,
                                 association_threshold=0.0)
    first = _scenario_person(180)
    moved = _scenario_person(260)
    distractor = _scenario_person(340)
    crossed = _scenario_person(290)
    distractor_after = _scenario_person(210)
    crossing = [tracker.update(_scenario_face(first), [first, _scenario_person(420)], (480, 640, 3)).state.value,
                tracker.update(_scenario_face(distractor, confidence=0.5), [moved, distractor], (480, 640, 3)).state.value,
                tracker.update(_scenario_face(crossed), [crossed, distractor_after], (480, 640, 3)).state.value]
    first_frame = np.zeros((160, 160, 3), dtype=np.uint8)
    second_frame = np.zeros_like(first_frame)
    cv2.rectangle(first_frame, (40, 40), (90, 130), (255, 255, 255), -1)
    cv2.rectangle(second_frame, (46, 40), (96, 130), (255, 255, 255), -1)
    body_tracker = BodyContinuityTracker(max_missing=3)
    seeded = body_tracker.start(first_frame, PersonDetection(np.array([40, 40, 90, 130], dtype=np.float32)))
    predicted = body_tracker.predict(second_frame)
    body_dropout = ["Confirmed", "Temporary"] if seeded and predicted is not None else ["Confirmed", "Unknown"]
    return {"head_turn": {"states": head_turn, "passed": head_turn[:2] == ["Confirmed", "Confirmed"]
                           and "Temporary" in head_turn and head_turn[-1] == "Confirmed"},
            "occlusion": {"states": occlusion, "passed": occlusion[0] == "Confirmed" and "Unknown" in occlusion},
            "path_crossing": {"states": crossing, "passed": crossing == ["Confirmed", "Temporary", "Confirmed"]},
            "body_detector_dropout": {"states": body_dropout, "passed": body_dropout == ["Confirmed", "Temporary"]}}


def _live(args: argparse.Namespace) -> None:
    from .database import FaceDatabase
    from .embed import ArcFace
    from .landmarks import FaceDetector
    from .workflow import camera, exiting, read_frame

    detector = FaceDetector(args.detector, args.face_confidence)
    embedder = ArcFace(args.model, args.preprocessing)
    db = FaceDatabase(args.db, embedder.signature)
    if args.target not in db.people:
        raise ValueError(f"Target {args.target!r} is not enrolled; run src.enroll first")
    person_detector = HOGPersonDetector()
    pose_detector = PosePersonDetector()
    continuity = BodyContinuityTracker(args.max_temporary_frames)
    def identity(face: Face) -> tuple[bool, Optional[float]]:
        # Keep small detections visible to the temporal tracker, but do not
        # treat an upscaled, low-detail crop as reliable identity evidence.
        if min(face.box[2:4]) < args.min_face_size:
            return False, None
        try:
            vector = embedder.embed(align_face(current_frame[0], face.points))
            name, score = db.match(vector, args.threshold, args.margin)
            return name == args.target, score
        except ValueError:
            return False, None

    tracker = WholePersonTracker(identity, decay_rate=args.decay_rate,
                                 temporary_floor=args.temporary_floor,
                                 max_temporary_frames=args.max_temporary_frames,
                                 dead_zone=args.dead_zone,
                                 face_mismatch_grace=args.face_mismatch_grace)
    current_frame = [None]
    window = "FaceX whole-person tracking"
    try:
        with camera(args.camera) as cap:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while True:
                frame = read_frame(cap)
                current_frame[0] = frame
                # Preserve every YuNet detection for the body lock. Faces below
                # the ArcFace size floor count as uncertain, not as a sudden
                # disappearance; identity() rejects their embeddings above.
                faces = detector.detect(frame)
                hog_persons = person_detector.detect(frame)
                pose_persons = pose_detector.detect(frame)
                detected_persons = merge_person_detections(hog_persons, pose_persons)
                persons = add_face_guided_fallbacks(faces, detected_persons, frame.shape)
                predicted = continuity.predict(frame)
                # A predicted body is only needed during facial loss. Keeping it
                # out of face-present frames avoids competing with fresh geometry.
                if not faces and predicted is not None:
                    persons.append(predicted)
                result = tracker.update(faces, persons, frame.shape)
                if result.person is not None and any(result.person is item for item in persons):
                    if continuity.last_box is None or predicted is None or result.person is not predicted:
                        continuity.start(frame, result.person)
                elif result.state == IdentityState.UNKNOWN and result.person is None:
                    continuity.reset()
                draw_result(frame, result, args.target)
                for face in faces:
                    if face is result.face:
                        continue
                    fx, fy, fw, fh = np.rint(face.box).astype(int)
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 190, 0), 1)
                    cv2.putText(frame, "face detected", (max(0, fx), max(16, fy - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"faces={len(faces)} bodies={len(persons)} "
                                  f"hog={len(hog_persons)} pose={len(pose_persons)} "
                                  f"motion={'yes' if predicted is not None else 'no'}",
                            (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)
                if exiting(window, cv2.waitKey(1) & 0xff):
                    break
    finally:
        pose_detector.close()


def main() -> None:
    # Parse the model-free command before importing workflow/ONNX dependencies.
    if "--evaluate" in sys.argv:
        cli = argparse.ArgumentParser(description=__doc__)
        cli.add_argument("--evaluate", action="store_true")
        cli.parse_args()
        report = evaluate_scenarios()
        for name, data in report.items():
            print(f"{name}: {'PASS' if data['passed'] else 'FAIL'} | states={data['states']}")
        return

    from .workflow import parser
    cli = parser(__doc__)
    cli.add_argument("--target", help="Enrolled identity to follow in webcam mode")
    cli.add_argument("--evaluate", action="store_true", help="Run deterministic scenarios and exit")
    cli.add_argument("--threshold", type=float, default=0.45)
    cli.add_argument("--margin", type=float, default=0.05)
    cli.add_argument("--min-face-size", type=int, default=30)
    cli.add_argument("--face-confidence", type=float, default=0.60,
                     help="YuNet confidence cutoff for live faces")
    cli.add_argument("--decay-rate", type=float, default=0.08)
    cli.add_argument("--temporary-floor", type=float, default=0.05)
    cli.add_argument("--max-temporary-frames", type=int, default=24)
    cli.add_argument("--face-mismatch-grace", type=int, default=3,
                     help="Consecutive visible-face match misses tolerated before Temporary")
    cli.add_argument("--dead-zone", type=float, default=0.08)
    args = cli.parse_args()
    # Keep the whole-person command aligned with the project's documented
    # webcam default. An explicit flag always wins for multi-camera setups.
    if "--camera" not in sys.argv:
        args.camera = 0
    if not 0 < args.face_confidence <= 1:
        cli.error("--face-confidence must be in (0, 1]")
    if not args.target:
        cli.error("--target is required for webcam mode; use --evaluate for model-free scenarios")
    _live(args)


if __name__ == "__main__":
    # Keep the deterministic evaluator usable in a minimal CPU environment
    # without importing the optional ONNX runtime needed by webcam mode.
    if "--evaluate" in sys.argv:
        main()
    else:
        from .workflow import run
        run(main)
