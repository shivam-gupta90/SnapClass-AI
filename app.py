import streamlit as st
from src.screens.home_screen import home_screen
from src.screens.student_screen import student_screen
from src.screens.teacher_screen import teacher_screen
from src.components.dialog_auto_enroll import auto_enroll_dialog

def main():
    st.set_page_config(
        page_title="SnapClass - Making Attendance faster using AI",
        page_icon="https://i.ibb.co/YTYGn5qV/logo.png"
    )

    if "login_type" not in st.session_state:
        st.session_state["login_type"] = None

    # Read join code
    join_code = st.query_params.get("join_code")

    # If opened using invite link, go directly to student login
    if join_code and st.session_state["login_type"] is None:
        st.session_state["login_type"] = "student"
        st.rerun()

    # If student is already logged in and join code exists,
    # open dialog BEFORE rendering dashboard
    if (
        join_code
        and st.session_state.get("is_logged_in")
        and st.session_state.get("user_role") == "student"
    ):
        auto_enroll_dialog(join_code)

    match st.session_state["login_type"]:
        case "teacher":
            teacher_screen()

        case "student":
            student_screen()

        case _:
            home_screen()

if __name__ == "__main__":
    main()