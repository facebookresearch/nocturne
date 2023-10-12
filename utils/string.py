"""Collection of utilities for strings."""
from datetime import date, datetime


def datetime_to_str(dt: datetime) -> str:
    """Converts a datetime object to a string.

    Args:
        dt (datetime): Datetime object to convert.

    Returns:
        str: String representation of the datetime object.
    """
    return dt.strftime("%Y_%m_%d__%H_%M_%S")


def date_to_str(date_: date) -> str:
    """Converts a date object to a string.

    Args:
        date_ (date): Date object to convert.

    Returns:
        str: String representation of the date object.
    """
    return date_.strftime("%Y_%m_%d")