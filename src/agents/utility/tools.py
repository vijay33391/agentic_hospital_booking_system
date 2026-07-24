import os
import re
from datetime import datetime
from typing import Literal

from src.agents.utility.validators import (
    DateValidator,
    DateTimeModel,
    IdentifiactionNumberValidator,
    DOCTOR_NAMES,)
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
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
# create table public.adv_booking_system (
#   date_slot timestamp without time zone null,
#   specialization character varying null,
#   doctor_name character varying null,
#   is_available boolean null,
#   patient_to_attend bigint null
# ) TABLESPACE pg_default;
#
# NOTE: this table has no primary key / id column, so all UPDATE/lookup
# queries match on the (doctor_name, date_slot) pair instead of an id.


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool(
    "check_doctor_availability",
    description="Check the doctor's availability based on the given date.",
)
def check_doctor_availability(
    desired_date: DateValidator,
    doctor_name: DOCTOR_NAMES,
) -> str:
    """
    Check the availability of a doctor on a given date.

    Args:
        desired_date (DateValidator): Date in YYYY-MM-DD format.
        doctor_name (Literal): Name of the doctor.

    Returns:
        str: Available appointment slots in AM/PM format or a message if none exist.
    """
    query = text("""
        SELECT date_slot
        FROM adv_booking_system
        WHERE doctor_name = :doctor_name
          AND DATE(date_slot) = :appointment_date
          AND is_available = TRUE
        ORDER BY date_slot;
    """)

    try:
        # desired_date.date is already validated as YYYY-MM-DD by DateValidator.
        appointment_date = datetime.strptime(desired_date.date, "%Y-%m-%d").date()

        with SessionLocal() as session:
            rows = session.execute(
                query,
                {"doctor_name": doctor_name, "appointment_date": appointment_date},
            ).fetchall()

        if not rows:
            return f"No available slots found for {doctor_name.title()} on {desired_date.date}."

        slots = [row.date_slot.strftime("%I:%M %p") for row in rows]
        return (
            f"Available slots for {doctor_name.title()} on {desired_date.date}:\n\n"
            + "\n".join(slots)
        )

    except Exception as e:
        return f"Database Error: {str(e)}"


@tool('check_doctor_availability_by_specialization',description="Check the availability of doctors by specialization on a given date.")
def check_doctor_availability_by_specialization(
    desired_date: DateValidator,
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
        desired_date (DateValidator): Date in YYYY-MM-DD format.
        specialization (Literal): Doctor specialization.

    Returns:
        str: Available doctors and their slots.
    """
    query = text("""
        SELECT doctor_name, date_slot
        FROM adv_booking_system
        WHERE specialization = :specialization
          AND DATE(date_slot) = :appointment_date
          AND is_available = TRUE
        ORDER BY doctor_name, date_slot;
    """)

    try:
        appointment_date = datetime.strptime(desired_date.date, "%Y-%m-%d").date()

        with SessionLocal() as session:
            rows = session.execute(
                query,
                {"specialization": specialization, "appointment_date": appointment_date},
            ).fetchall()

        if not rows:
            return (
                f"No available {specialization.replace('_', ' ')} "
                f"doctors found on {desired_date.date}."
            )

        response = (
            f"Available {specialization.replace('_', ' ').title()} "
            f"doctors on {desired_date.date}:\n\n"
        )
        for row in rows:
            response += f"• {row.doctor_name.title()} - {row.date_slot.strftime('%I:%M %p')}\n"

        return response

    except Exception as e:
        return f"Database Error: {str(e)}"


@tool("cancel_appointment",description="Cancel an existing appointment.")
def cancel_appointment(
    desired_date: DateTimeModel,
    id_number: IdentifiactionNumberValidator,
    doctor_name: DOCTOR_NAMES,
) -> str:
    """
    Cancel an existing appointment.
    """
    with SessionLocal() as session:
        try:
            # desired_date.date_time is already validated as YYYY-MM-DD HH:MM:SS
            # by DateTimeModel.
            appointment_datetime = datetime.strptime(
                desired_date.date_time, "%Y-%m-%d %H:%M:%S"
            )

            cancel_query = text("""
                UPDATE adv_booking_system
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
                    "patient_id": id_number.id_number,
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


@tool('set_appointment',description="Check  the available slots. it is available book appointment slot with the doctor.")
def set_appointment(
    desired_date: DateTimeModel,
    id_number: IdentifiactionNumberValidator,
    doctor_name: DOCTOR_NAMES,
) -> str:
    """
        Check  the available slots. it is available Set appointment slot with the doctor.
        
    """
    with SessionLocal() as session:
        try:
            appointment_datetime = datetime.strptime(
                desired_date.date_time, "%Y-%m-%d %H:%M:%S"
            )

            # adv_booking_system has no id column, so we match/lock on
            # doctor_name + date_slot directly.
            book_query = text("""
                UPDATE adv_booking_system
                SET
                    is_available = FALSE,
                    patient_to_attend = :patient_id
                WHERE doctor_name = :doctor_name
                  AND date_slot = :appointment_datetime
                  AND is_available = TRUE
                RETURNING date_slot;
            """)

            booked = session.execute(
                book_query,
                {
                    "patient_id": id_number.id_number,
                    "doctor_name": doctor_name,
                    "appointment_datetime": appointment_datetime,
                },
            ).fetchone()

            if booked is None:
                session.rollback()
                return (
                    "The requested appointment slot is unavailable "
                    "or has already been booked."
                )

            session.commit()

            return (
                f"Appointment confirmed with "
                f"{doctor_name.title()} on "
                f"{booked.date_slot.strftime('%d-%m-%Y %I:%M %p')}."
            )

        except Exception as e:
            session.rollback()
            return f"Failed to book appointment: {str(e)}"


@tool("reschedule_appointment",description="Reschedule an existing appointment.")
def reschedule_appointment(
    desired_date: DateTimeModel,
    id_number: IdentifiactionNumberValidator,
    doctor_name: DOCTOR_NAMES,
    new_date: DateTimeModel,
) -> str:
    """
    Reschedule an existing appointment.

    Args:
        desired_date: Current appointment date/time (YYYY-MM-DD HH:MM:SS)
        id_number: Patient identification number
        doctor_name: Doctor's name
        new_date: New appointment date/time (YYYY-MM-DD HH:MM:SS)

    Returns:
        str: Confirmation message
    """
    with SessionLocal() as session:
        try:
            current_datetime = datetime.strptime(
                desired_date.date_time, "%Y-%m-%d %H:%M:%S"
            )
            new_datetime = datetime.strptime(
                new_date.date_time, "%Y-%m-%d %H:%M:%S"
            )

            # adv_booking_system has no id column, so we match on
            # doctor_name + date_slot directly.

            # Step 1: Verify current appointment exists
            current_query = text("""
                SELECT date_slot
                FROM adv_booking_system
                WHERE doctor_name = :doctor_name
                  AND date_slot = :current_datetime
                  AND patient_to_attend = :patient_id;
            """)

            current_slot = session.execute(
                current_query,
                {
                    "doctor_name": doctor_name,
                    "current_datetime": current_datetime,
                    "patient_id": id_number.id_number,
                },
            ).fetchone()

            if current_slot is None:
                return "No appointment found to reschedule."

            # Step 2: Verify new slot is available
            new_slot_query = text("""
                SELECT date_slot
                FROM adv_booking_system
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
                    UPDATE adv_booking_system
                    SET
                        is_available = TRUE,
                        patient_to_attend = NULL
                    WHERE doctor_name = :doctor_name
                      AND date_slot = :old_datetime;
                """),
                {"doctor_name": doctor_name, "old_datetime": current_datetime},
            )

            # Step 4: Book new slot
            session.execute(
                text("""
                    UPDATE adv_booking_system
                    SET
                        is_available = FALSE,
                        patient_to_attend = :patient_id
                    WHERE doctor_name = :doctor_name
                      AND date_slot = :new_datetime;
                """),
                {
                    "patient_id": id_number.id_number,
                    "doctor_name": doctor_name,
                    "new_datetime": new_datetime,
                },
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
