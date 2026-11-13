"""
Surveillance Module for Real-Time Crowd Counting, Localization, Spatial Analytics & Notifications.
"""

from .stream import ThreadedCameraStream
from .detector import CrowdDetector
from .zones import SurveillanceZone, ZoneManager
from .tracker import KalmanPointTracker, TrackedPerson
from .tripwire import VirtualTripwire, TripwireManager
from .database import SurveillanceDatabase
from .notifier import NotificationManager
from .alerts import AlertRuleEngine

__all__ = [
    "ThreadedCameraStream",
    "CrowdDetector",
    "SurveillanceZone",
    "ZoneManager",
    "KalmanPointTracker",
    "TrackedPerson",
    "VirtualTripwire",
    "TripwireManager",
    "SurveillanceDatabase",
    "NotificationManager",
    "AlertRuleEngine",
]
