# Required Libraries
import streamlit as st
import torch
import pandas as pd
from model_utils import load_model, load_tokenizer, get_model_output
from utils import parse_input_string, extract_tables_from_docx, convert_docx_table_to_model_input, \
    display_text_word_by_word, generate_table_labels, table_to_index
import base64
import re
import time


def load_css():
    st.markdown("""
        <style>
        .paragraph-spacing p {
            margin-bottom: 10px;
        }
        .big-font {
            font-size:24px !important;
            /* line-height: 3; */
            margin-bottom: 40px;
        }
        .big-font-single {
            font-size:20px !important;
            padding: 10px;
        }
        .big-font-right {
            font-size:20px !important;
            text-align:right !important;
        }
        .section-header-black {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: white;
        background-color: black;
        
        }
        .section-header-grey {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: black;
        background-color: #EEEEEE;
        padding: 10px;
        }
        .section-subheader-black {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: white;
        background-color: black;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_data
def cached_load_model(model_dir):
    return load_model(model_dir)


@st.cache_data
def cached_load_tokenizer(model_dir):
    return load_tokenizer(model_dir)


def introduction_and_methodology_section():
    # Introduction and Methodology
    col1, col2 = st.columns([5,5])
    # Introduction section
    with col1:
        load_css()
        st.markdown('<div class="section-header-black">Welcome to AutoGen</div>', unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        col_icon, col_text = st.columns([0.5,9.5])
        with col_icon:
            st.image("icons/icon-easy.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        <b>Automatically Generated Table Summaries:</b> T-Gen automatically generate table summaries, 
                        obviating the requirement for manual interpretation and description.
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5,9.5])
        with col_icon:
            st.image("icons/icon-ai.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        <b>In-house Language Model:</b> T-Gen benefits from the state-of-the-art advancements in natural language processing
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5,9.5])
        with col_icon:
            st.image("icons/icon-convenient.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        <b>Streamlined Model Performance Analysis:</b> Integrate with AutoDoc enabling users quickly 
                        and effortlessly documentation experience.
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5,9.5])
        with col_icon:
            st.image("icons/icon-transformation.png")
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        <b>Customizable and Adaptable:</b> T-Gen can be further developed for other specific domains or datasets
                        </div>
                        """, unsafe_allow_html=True)

    with col2:
        load_css()
        st.markdown("""
            <style>
            .column1-content {
                background-color: #EEEEEE;
            }
            </style>
            <div class="section-header-grey">State-of-the-art</div>
            <div class="column1-content">
                <img src="https://github.com/MinhongW/text_generation/blob/main/imgs/fig_transformers_ver4_01.png?raw=true" width="850"/>
            </div>
            """, unsafe_allow_html=True)


def t5_model_section():
    # st.header("T5: Text-To-Text Transfer Transformer")
    col1, col2 = st.columns([5, 5])
    with col1:
        load_css()
        st.markdown("""
                    <style>
                    .column1-content {
                        background-color: #696969;
                    }
                    </style>
                    <div class="section-header-grey">Architecture</div>
                    <div class="column1-content">
                        <img src="https://github.com/MinhongW/text_generation/blob/main/imgs/fig_transformers_ver4.1_02.png?raw=true" width="850"/>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        load_css()
        st.markdown('<div class="section-header-black">T5: Text-To-Text Transfer Transformer</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-open-source.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                    <b>Open-source</b> language model developed by Google Research Lab
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-documents.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        Trained on a <b>750 GB</b> dataset of clean natural English free text
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-idea.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        An encoder-decoder model pre-trained on a multi-task mixture of tasks
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-options.png")
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        T5 provides 5 language models of multiple sizes, from 60M parameters to 11B
                        </div>
                        """, unsafe_allow_html=True)


def model_development_section():
    col1 = st.columns(1)[0]
    with col1:
        load_css()
        st.markdown("""
                            <style>
                            .column1-content {
                                background-color: #EEEEEE;
                            }
                            </style>
                            <div class="section-header-grey">Model Development</div>
                            <div class="column1-content">
                                <img src="https://github.com/MinhongW/text_generation/blob/main/imgs/fig_pipeline_ver4.png?raw=true" width="1500"/>
                            </div>
                            """, unsafe_allow_html=True)


def model_section():
    # Model section
    st.markdown('<div class="section-header-black">AutoGen</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    model_output = None

    with col1:
        st.markdown('<div class="section-subheader-black">Input Config</div>',
                    unsafe_allow_html=True)
        # model_dir = st.text_input("Model Directory", "ca-rnd-language-model-small")
        col_model_dir, col_input_format = st.columns([5,5])
        with col_model_dir:
            model_dir = st.radio("Model name:",
                                     ("ca-rnd-language-model-small",
                                      "ca-rnd-language-model-medium",
                                      "ca-rnd-language-model-large"),
                                 )
        with col_input_format:
            input_format = st.radio("Choose an input format", ["Text", "Word document"])

        if model_dir != "ca-rnd-language-model-small":
            st.warning('Model is coming soon. Please select the small model for now.', icon="⚠️")

        else:
            model = cached_load_model(model_dir)
            tokenizer = cached_load_tokenizer(model_dir)

            # input_format = st.radio("Choose an input format", ["Text", "Word document"])

            if input_format == "Text":

                user_input = st.text_area("Input Text", "", height=200)

                if user_input:
                    caption, df, model_input = parse_input_string(user_input)
                    if df is not None:
                        st.markdown(f"**Caption: {caption}**")
                        st.dataframe(df)
                    else:
                        st.warning("Could not parse input as a table. Please check your input.")

            elif input_format == "Word document":

                uploaded_file = st.file_uploader("Upload a Word Document", type="docx")
                if uploaded_file is not None:
                    captions, tables = extract_tables_from_docx(uploaded_file)
                    # selected_table_index = st.selectbox("Choose a table", range(len(tables)))
                    table_labels = generate_table_labels(len(tables))
                    selected_table = st.selectbox("Choose a table", table_labels)
                    selected_table_index = table_to_index(selected_table)
                    st.markdown(f"**{captions[selected_table_index]}**")
                    st.dataframe(tables[selected_table_index])
                    model_input = convert_docx_table_to_model_input(captions[selected_table_index],
                                                                    tables[selected_table_index])

        with col2:
            st.markdown('<div class="section-subheader-black">Model Output</div>',
                        unsafe_allow_html=True)
            if st.button('Generate Output from Table'):
                with st.spinner("Generating output..."):
                    # Modify this line to handle table input appropriately
                    # st.markdown(f"**{model_input}**")
                    model_output = get_model_output(model_input, model, tokenizer)
                    # st.text_area("Model Output", model_output, height=300)
            if model_output:
                model_output = re.sub(r"<[^>]+>", "", model_output).lstrip()
                display_text_word_by_word(model_output)
                # st.text_area("", model_output, height=300)


def result_section():
    # st.markdown('<div class="section-header-black">AutoGen Performance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 5])
    # Introduction section
    with col1:
        load_css()
        st.markdown("""
                    <style>
                    .column1-content {
                        background-color: #EEEEEE;
                    }
                    </style>
                    <div class="section-header-grey">AutoGen Performance</div>
                    <div class="column1-content">
                        <img src="https://github.com/MinhongW/text_generation/blob/main/imgs/fig_results_ver04.png?raw=true" width="850"/>
                    </div>
                    """, unsafe_allow_html=True)

    # Methodology section
    with col2:
        # st.image("fig_results02.png",
        #          use_column_width=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-evaluation.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        <b>ROUGE</b> score measures the overlap of n-grams between two texts.
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-analytics.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        The baseline model achieved ROUGE-L as 29.70 by fine-tuning T5 model to generate scientific 
                        numeric tables' descriptions
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-increase.png", use_column_width=True)
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        Our model achieved <b>higher</b> ROUGE score (47.40) for the task of generating descriptions 
                        for tables describing model performance
                        </div>
                        """, unsafe_allow_html=True)
        col_icon, col_text = st.columns([0.5, 9.5])
        with col_icon:
            st.image("icons/icon-potential.png")
        with col_text:
            st.markdown("""
                        <div class="big-font">
                        The performance could potentially be increased by adapting bigger models
                        </div>
                        """, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide")

    load_css()

    col1, col2 = st.columns([8, 2])

    with col1:
        st.title("T-Gen: A Large Language Model for Text Generation")

    with col2:
        st.markdown("""
                    <div class="big-font-right">
                    RnD, Compliance Analytics <br>
                    Developed by <br>
                    Minhong Wang, Data Scientist <br>
                    Hassan Saif, Senior Manager <br>
                    </div>
                """, unsafe_allow_html=True)
    #st.write("")
    st.markdown("""
            <br><br><br>
            """, unsafe_allow_html=True)
    #st.write("")
    #st.write("")
    #st.write("")
    introduction_and_methodology_section()
    t5_model_section()
    model_development_section()
    model_section()
    result_section()

    st.markdown("---")  # Horizontal line
    st.markdown("""
        Research and Development, Compliance Analytics, HSBC.
        
        For more information, please contact Minhong Wang at minhong.wang@hsbc.com.
        
        """)


if __name__ == "__main__":
    main()
