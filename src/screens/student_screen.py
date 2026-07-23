import streamlit as st
from src.ui.base_layout import style_background_dashboard,style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
import numpy as np
from src.pipelines.face_pipeline  import predict_attendance, get_face_embeddings,train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
from src.database.db import get_all_students,create_student
import time

def student_dashboard():
    st.header("Dashboard Here")
def student_screen():

    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return


    c1,c2 = st.columns(2,vertical_alignment='center', gap='xxlarge')

    with c1:
        header_dashboard()
    with c2:
        if st.button("Go to Back Home", type='secondary',key='loginbackbtn',shortcut="control+backspace"):
            st.session_state['login_type']= None
            st.rerun()
    st.header('Login using FaceID',text_alignment='center')

    st.space()
    st.space()

    show_registration = False

    photo_source = st.camera_input("Position your Face in the Center")
    if photo_source:
        img = np.array(Image.open(photo_source))

        with st.spinner("AI is Scanning..."):
            detected, all_ids, num_faces=predict_attendance(img)

            if num_faces ==0:
                st.warning('Face not Found!')
            elif num_faces>2:
                st.warning('Multiple Faces Found')
            else:
                if detected:
                    student_id = list(detected.keys())[0]
                    all_students = get_all_students()
                    student = next((s for s in all_students if s['student_id']==student_id),None)
                    
                    if student:
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = 'student'
                        st.session_state.student_data = student
                        st.toast(f"Welcome Back {student['name']}")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.info("Face not recognized! You might be new student!")
                    show_registration =True
    if show_registration:
        with st.container(border=True):
            st.header('Register New Profile')
            new_name = st.text_input("Enter your Name",placeholder='E.g. Kapil Kumar')

            st.subheader('Optional : Voice Enrollment')
            st.info('Enroll your For Voice  only attendace')

            audio_data = None

            try:
                audio_data = st.audio_input('Record a short phrase like I am present, My name is kapil')
            except Exception:
                st.error('Audio Data Failed')

            if st.button('Create Account',type='primary'):
                if new_name:
                    with st.spinner('Creating Profile...'):
                        img = np.array(Image.open(photo_source))
                        encoding = get_face_embeddings(img)
                        if encoding:
                            face_emb = encoding[0].tolist()

                            voice_emb = None
                            if audio_data:
                                voice_emb = get_voice_embedding(audio_data.read())
                            
                            response_data = create_student(new_name,face_embedding = face_emb,voice_embedding= voice_emb)

                            if response_data:
                                train_classifier()
                                st.session_state.is_logged_in = True
                                st.session_state.user_role = 'student'
                                st.session_state.student_data = response_data[0]
                                st.toast(f'Profile Created! Hi {new_name}')
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.error('Coudnot Capture your Facial Feature For Registration')
                
                else:
                    st.warning('Please Enter Your Name')




    footer_dashboard()