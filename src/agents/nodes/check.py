import os
import re
from datetime import datetime
from typing import Literal

from src.agents.utility.validators import (
    DateValidator,
    DateTimeModel,
    IdentifiactionNumberValidator,
    DOCTOR_NAMES,
)
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field, field_validator, ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain.tools import tool


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
SUPABASE_URI = os.getenv("SUPABASE_URI")
if not SUPABASE_URI:
    raise EnvironmentError(
        "SUPABASE_URI environment variable is not set. "
        "Add it to your .env file before starting the app."
    )

engine = create_engine(url=SUPABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Reference schema (for documentation only):
# create table public.doctor_availability (
#   id bigserial not null,
#   date_slot timestamp without time zone not null,
#   specialization character varying(100) not null,
#   doctor_name character varying(100) not null,
#   is_available boolean not null,
#   patient_to_attend bigint null,
#   constraint doctor_availability_pkey primary key (id)
# );


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
# Tool parameters below are plain str/int, NOT the Pydantic validator classes
# directly. When a tool parameter's type is itself a BaseModel, the JSON
# schema generated for the LLM requires a *nested object* for that argument
# (e.g. {"desired_date": {"date_time": "..."}}). Groq's tool-calling with
# llama-3.3-70b-versatile reliably fails to construct that nesting — it was
# sending a flattened `date_time` key and a bare int for `id_number` instead
# of the required nested objects, which the API then rejected outright.
#
# Flattening the parameters to plain types fixes the tool-calling schema.
# We still validate the same way as before, just inside the function body,
# via these small helpers that turn a ValidationError into a clear string
# the agent can relay back to the user instead of crashing.

def _validate_date(desired_date: str) -> tuple[datetime.date | None, str | None]:
    try:
        model = DateValidator(date=desired_date)
    except ValidationError as e:
        return None, f"Invalid date '{desired_date}': {e.errors()[0]['msg']}"
    return datetime.strptime(model.date, "%d-%m-%Y").date(), None


def _validate_date_time(desired_date_time: str) -> tuple[datetime | None, str | None]:
    try:
        model = DateTimeModel(date_time=desired_date_time)
    except ValidationError as e:
        return None, f"Invalid date/time '{desired_date_time}': {e.errors()[0]['msg']}"
    return datetime.strptime(model.date_time, "%d-%m-%Y %H:%M"), None


def _validate_id_number(id_number: int) -> tuple[int | None, str | None]:
    try:
        model = IdentifiactionNumberValidator(id_number=id_number)
    except ValidationError as e:
        return None, f"Invalid ID number '{id_number}': {e.errors()[0]['msg']}"
    return model.id_number, None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool(
    "check_doctor_availability",
    description="Check the doctor's availability based on the given date.",
)
def check_doctor_availability(
    desired_date: str,
    doctor_name: DOCTOR_NAMES,
) -> str:
    """
    Check the availability of a doctor on a given date.

    Args:
        desired_date (str): Date in DD-MM-YYYY format.
        doctor_name (Literal): Name of the doctor.

    Returns:
        str: Available appointment slots in AM/PM format or a message if none exist.
    """
    appointment_date, error = _validate_date(desired_date)
    if error:
        return error

    query = text("""
        SELECT id, date_slot
        FROM doctor_availability
        WHERE doctor_name = :doctor_name
          AND DATE(date_slot) = :appointment_date
          AND is_available = TRUE
        ORDER BY date_slot;
    """)

    try:
        with SessionLocal() as session:
            rows = session.execute(
                query,
                {"doctor_name": doctor_name, "appointment_date": appointment_date},
            ).fetchall()

        if not rows:
            return f"No available slots found for {doctor_name.title()} on {desired_date}."

        slots = [row.date_slot.strftime("%I:%M %p") for row in rows]
        return (
            f"Available slots for {doctor_name.title()} on {desired_date}:\n\n"
            + "\n".join(slots)
        )

    except Exception as e:
        return f"Database Error: {str(e)}"


@tool(
    "check_doctor_availability_by_specialization",
    description="Check the availability of doctors by specialization on a given date.",
)
def check_doctor_availability_by_specialization(
    desired_date: str,
    specialization: Literal[
        "general_dentist",
        "cosmetic_dentist",
        "prosthodontist",
        "pediatric_dentist",
        "emergency_dentist",
        "oral_surgeon",
        "orthodontist",
    ],
) -> str:
    """
    Check the availability of doctors by specialization on a given date.

    Args:
        desired_date (str): Date in DD-MM-YYYY format.
        specialization (Literal): Doctor specialization.

    Returns:
        str: Available doctors and their slots.
    """
    appointment_date, error = _validate_date(desired_date)
    if error:
        return error

    query = text("""
        SELECT doctor_name, date_slot
        FROM doctor_availability
        WHERE specialization = :specialization
          AND DATE(date_slot) = :appointment_date
          AND is_available = TRUE
        ORDER BY doctor_name, date_slot;
    """)

    try:
        with SessionLocal() as session:
            rows = session.execute(
                query,
                {"specialization": specialization, "appointment_date": appointment_date},
            ).fetchall()

        if not rows:
            return (
                f"No available {specialization.replace('_', ' ')} "
                f"doctors found on {desired_date}."
            )

        response = (
            f"Available {specialization.replace('_', ' ').title()} "
            f"doctors on {desired_date}:\n\n"
        )
        for row in rows:
            response += f"• {row.doctor_name.title()} - {row.date_slot.strftime('%I:%M %p')}\n"

        return response

    except Exception as e:
        return f"Database Error: {str(e)}"


@tool("cancel_appointment", description="Cancel an existing appointment.")
def cancel_appointment(
    desired_date: str,
    id_number: int,
    doctor_name: DOCTOR_NAMES,
) -> str:
    """
    Cancel an existing appointment.

    Args:
        desired_date (str): Existing appointment date/time in DD-MM-YYYY HH:MM format.
        id_number (int): Patient identification number (7-8 digits).
        doctor_name (Literal): Doctor's name.
    """
    appointment_datetime, error = _validate_date_time(desired_date)
    if error:
        return error

    patient_id, error = _validate_id_number(id_number)
    if error:
        return error

    with SessionLocal() as session:
        try:
            cancel_query = text("""
                UPDATE doctor_availability
                SET
                    is_available = TRUE,
                    patient_to_attend = NULL
                WHERE doctor_name = :doctor_name
                  AND date_slot = :appointment_datetime
                  AND patient_to_attend = :patient_id
                RETURNING date_slot;
            """)

            cancelled = session.execute(
                cancel_query,
                {
                    "doctor_name": doctor_name,
                    "appointment_datetime": appointment_datetime,
                    "patient_id": patient_id,
                },
            ).fetchone()

            if cancelled is None:
                session.rollback()
                return (
                    "No matching appointment was found. "
                    "Please verify the doctor, date/time, and patient ID."
                )

            session.commit()

            return (
                f"Appointment with {doctor_name.title()} on "
                f"{cancelled.date_slot.strftime('%d-%m-%Y %I:%M %p')} "
                f"has been cancelled successfully."
            )

        except Exception as e:
            session.rollback()
            return f"Failed to cancel appointment: {str(e)}"


@tool(
    "set_appointment",
    description="Check the available slots, and if available, set the appointment slot with the doctor.",
)
def set_appointment(
    desired_date: str,
    id_number: int,
    doctor_name: DOCTOR_NAMES,
) -> str:
    """
    Check the available slots and set an appointment slot with the doctor.

    Args:
        desired_date (str): Requested date/time in DD-MM-YYYY HH:MM format.
        id_number (int): Patient identification number (7-8 digits).
        doctor_name (Literal): Doctor's name.
    """
    appointment_datetime, error = _validate_date_time(desired_date)
    if error:
        return error

    patient_id, error = _validate_id_number(id_number)
    if error:
        return error

    with SessionLocal() as session:
        try:
            check_query = text("""
                SELECT id
                FROM doctor_availability
                WHERE doctor_name = :doctor_name
                  AND date_slot = :appointment_datetime
                  AND is_available = TRUE;
            """)

            slot = session.execute(
                check_query,
                {"doctor_name": doctor_name, "appointment_datetime": appointment_datetime},
            ).fetchone()

            if slot is None:
                return (
                    "The requested appointment slot is unavailable "
                    "or has already been booked."
                )

            book_query = text("""
                UPDATE doctor_availability
                SET
                    is_available = FALSE,
                    patient_to_attend = :patient_id
                WHERE id = :slot_id
                RETURNING date_slot;
            """)

            booked = session.execute(
                book_query,
                {"patient_id": patient_id, "slot_id": slot.id},
            ).fetchone()

            session.commit()

            return (
                f"Appointment confirmed with "
                # Fixed typo: was `booked.date_slotd` (extra 'd'), which
                # would raise AttributeError on every successful booking.
                f"{doctor_name.title()} on "
                f"{booked.date_slot.strftime('%d-%m-%Y %I:%M %p')}."
            )

        except Exception as e:
            session.rollback()
            return f"Failed to book appointment: {str(e)}"


@tool("reschedule_appointment", description="Reschedule an existing appointment.")
def reschedule_appointment(
    desired_date: str,
    id_number: int,
    doctor_name: DOCTOR_NAMES,
    new_date: str,
) -> str:
    """
    Reschedule an existing appointment.

    Args:
        desired_date (str): Current appointment date/time (DD-MM-YYYY HH:MM)
        id_number (int): Patient identification number (7-8 digits)
        doctor_name (Literal): Doctor's name
        new_date (str): New appointment date/time (DD-MM-YYYY HH:MM)

    Returns:
        str: Confirmation message
    """
    current_datetime, error = _validate_date_time(desired_date)
    if error:
        return error

    new_datetime, error = _validate_date_time(new_date)
    if error:
        return error

    patient_id, error = _validate_id_number(id_number)
    if error:
        return error

    with SessionLocal() as session:
        try:
            # Step 1: Verify current appointment exists
            current_query = text("""
                SELECT id
                FROM doctor_availability
                WHERE doctor_name = :doctor_name
                  AND date_slot = :current_datetime
                  AND patient_to_attend = :patient_id;
            """)

            current_slot = session.execute(
                current_query,
                {
                    "doctor_name": doctor_name,
                    "current_datetime": current_datetime,
                    "patient_id": patient_id,
                },
            ).fetchone()

            if current_slot is None:
                return "No appointment found to reschedule."

            # Step 2: Verify new slot is available
            new_slot_query = text("""
                SELECT id
                FROM doctor_availability
                WHERE doctor_name = :doctor_name
                  AND date_slot = :new_datetime
                  AND is_available = TRUE;
            """)

            new_slot = session.execute(
                new_slot_query,
                {"doctor_name": doctor_name, "new_datetime": new_datetime},
            ).fetchone()

            if new_slot is None:
                return "The requested new appointment slot is unavailable."

            # Step 3: Free old slot
            session.execute(
                text("""
                    UPDATE doctor_availability
                    SET
                        is_available = TRUE,
                        patient_to_attend = NULL
                    WHERE id = :old_slot_id;
                """),
                {"old_slot_id": current_slot.id},
            )

            # Step 4: Book new slot
            session.execute(
                text("""
                    UPDATE doctor_availability
                    SET
                        is_available = FALSE,
                        patient_to_attend = :patient_id
                    WHERE id = :new_slot_id;
                """),
                {"patient_id": patient_id, "new_slot_id": new_slot.id},
            )

            session.commit()

            return (
                f"Appointment successfully rescheduled.\n\n"
                f"Doctor: {doctor_name.title()}\n"
                f"Old Slot: {current_datetime.strftime('%d-%m-%Y %I:%M %p')}\n"
                f"New Slot: {new_datetime.strftime('%d-%m-%Y %I:%M %p')}"
            )

        except Exception as e:
            session.rollback()
            return f"Failed to reschedule appointment: {str(e)}"