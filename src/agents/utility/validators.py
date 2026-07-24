
# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
import os
import re
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

import re
from datetime import datetime
from pydantic import BaseModel, Field, field_validator



class DateValidator(BaseModel):
    date: str = Field(
        description="Date in YYYY-MM-DD format"
    )

    @field_validator("date")
    def validate_date(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format")

        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"'{v}' is not a valid calendar date")

        return v


class DateTimeModel(BaseModel):
    date_time: str = Field(
        description="Date and time in YYYY-MM-DD HH:MI:SS format"
    )

    @field_validator("date_time")
    def validate_date_time(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", v):
            raise ValueError(
                "DateTime must be in YYYY-MM-DD HH:MI:SS format"
            )

        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(f"'{v}' is not a valid calendar date/time")

        return v

class IdentifiactionNumberValidator(BaseModel):
    id_number: int = Field(description="Identification number consisting of 7 to 8 digits")

    @field_validator("id_number")
    def id_validator(cls, v):
        """ID must contain only digits and be 7 to 8 digits long."""
        if not re.match(r"^\d{7,8}$", str(v)):
            raise ValueError("ID must be 7 to 8 digits")
        return v


DOCTOR_NAMES = Literal[
    "kevin anderson",
    "robert martinez",
    "susan davis",
    "daniel miller",
    "sarah wilson",
    "michael green",
    "lisa brown",
    "jane smith",
    "emily johnson",
    "john doe",
]