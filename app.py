import codecs
import json
import time
from io import StringIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from cachetools import LRUCache
from streamlit_modal import Modal

from utils import *


@st.cache_resource
def load_translator():
    return NmtTranslator(cfg)


@st.cache_resource
def set_cache():
    return LRUCache(maxsize=1024)


@st.cache_resource
def set_logs():
    return []

@st.cache_resource
def set_document():
    return []

@st.cache_resource
def convert_df(df):
    df.index += 1
    csv_data = df.to_csv(index_label='Index', index=True).encode('utf-8')
    csv_data_encoded = codecs.BOM_UTF8 + csv_data
    return csv_data_encoded


modal = Modal(key="logsKey", title="Logs")

cfg = {
    'batch_size': 64,
    'vocab_size': 65003,  # 65002 65003
    'max_seq_len': 50,  # 50 40
    'embedding_size': 1024,
    'num_heads': 16,  # 12 8
    'num_layers': 12,  # 6 12
    'pad_idx': 65000,

    # 'num_gpus': torch.cuda.device_count(),

    'src_pe_path': './vocab/source.spm',
    'tgt_pe_path': './vocab/target.spm',

    'vocab_path': './vocab/vocab.json',
    'checkpoint': './checkpoints/pytorch.safetensors'
}
cache = set_cache()
logs = set_logs()

translator = load_translator()
HCD_Access_key = "HCD_chaodu_ak_PUuSok01BRy9Nl5uSHnodtEtZw57qrlz"
access_key = None


def sidebar_widgets():
    # sidebar
    global access_key
    access_key = st.sidebar.text_input("Access Key", type="password")
    st.sidebar.title("Profile")
    st.sidebar.markdown("## " + "Project: ChaoDu Translator")
    st.sidebar.markdown(
        """
        <div style="text-align: right; font-family: Arial; font-size: 16px;">
         ——A machine translation platform
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.divider()
    st.sidebar.markdown("### " + "model:T5")
    st.sidebar.markdown(
        """
        <div font-family: Arial;">
         * Total params: 439,656,939<br>* Vocab Size: 65,000<br>* Seq Length: 50
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.divider()
    st.sidebar.markdown("### " + "Contributors: [HCD,LZX,ZT,ZQH]")

    st.sidebar.markdown("#### " + "Huang Chengdong:")
    st.sidebar.markdown(
        """
        * ##### model Training
        * ##### Inference API
        * ##### Speed Optimization
        """)

    st.sidebar.markdown("#### " + "Li Zhixun:")
    st.sidebar.markdown(
        """
        * ##### Front End
        """)

    st.sidebar.markdown("#### " + "Zhong Tao:")
    st.sidebar.markdown(
        """
        * ##### Back End
        """)

    st.sidebar.markdown("#### " + "Zhang Qihang:")
    st.sidebar.markdown(
        """
        * ##### Test Program
        """)


def authorize():
    return access_key == HCD_Access_key


def translate_click(msg,
                    input_texts,
                    input_ids,
                    target_language,
                    target_direction,
                    output_containers,
                    ):
    if authorize():
        num_input = len(input_texts)
        if num_input == 0:
            msg.error("Please put in the sentence you need to translate.", icon="🚨")
        else:
        # cache
            not_cached = []
            cached = []
            for idx in range(num_input):
                cached_text = cache.get((target_language, input_texts[idx]))
                if cached_text is None:
                    not_cached.append(input_texts[idx])
                else:
                    cached.append((idx, cached_text))

            start_time = time.time()

            input_time = time.asctime()

            progress_bar = st.progress(0.3, text="Translation in progress…")
                
            output_texts = translator.translate(not_cached, target_language)
                
            end_time = time.time()

            runtime = end_time - start_time
            while True:
                elapsed_time  = time.time()  - start_time
                progress_bar.progress(min(elapsed_time / runtime, 1.0), text="Translation in progress…")
                if elapsed_time >= runtime:
                    progress_bar.progress(min(elapsed_time / runtime, 1.0), text="Translation Complete!")
                    break

            # cached the not cached seq
            for idx in range(len(not_cached)):
                cache[(target_language, not_cached[idx])] = output_texts[idx]

            # merge cached and not cached
            for idx, cached_text in cached:
                output_texts.insert(idx, cached_text)

            for idx in range(num_input):
                output_containers[input_ids[idx]].markdown(
                    "```\n" + output_texts[idx] + "\n```",
                    unsafe_allow_html=True)

            for idx in range(num_input):
                logs.append([input_time, target_direction, input_texts[idx], output_texts[idx]])

            msg.success("Runtime:{:.4f}sec Seq Num:{} Avg:{:.4f}sec/seq Cache hits:{}"
                        .format(runtime, num_input, runtime / num_input, len(cached)),
                        icon="🎈")
    else:
        msg.error("Please input your Access Key", icon="🚨")
    
def logs_click(msg):
    if authorize():
        html_string = ''''''

        with modal.container():
            for idx in range(len(logs) - 1, -1, -1):
                html_string += '''
            <div class="logsContent">
                <div class="time"><strong>Time:</strong> ''' + str(logs[idx][0]) + '''</div>
                <div class="direction"><strong>Direction:</strong> ''' + str(logs[idx][1]) + '''</div>
                <div class="source"><strong>Source:</strong> ''' + str(logs[idx][2]) + '''</div>
                <div class="target"><strong>Target:</strong> ''' + str(logs[idx][3]) + '''</div>
                <br>''' + ''' 
            </div>
                <style>
                    .time, 
                    .source, 
                    .direction, 
                    .target {
                        margin-top:1px;
                        font-size: 14px;
                        font-family: Arial;
                    }
                </style>
                <script language="javascript"></script>
                '''
            components.html(html_string, height=550, scrolling=True)
    else:
        msg.error("Please input your Access Key", icon="🚨")

def home_body_content():
    st.title('ChaoDu Translator')
    msg = st.empty()
    import_button = st.button("Document Translate", type="primary", help="please double click.")
    if import_button:
        st.session_state.page = "secondPage"

    # 创建一个下拉菜单，用于选择目标语言
    target_language = st.selectbox('Choose Target Language', ['English', 'Chinese'])

    match_translator_api = {
        'English': 'en',
        'Chinese': 'zh'
    }
    translate_directions = {
        'English': 'ZH → EN',
        'Chinese': 'EN → ZH'
    }

    target_direction = translate_directions[target_language]
    target_language = match_translator_api[target_language]
    
    logs_csv = convert_df(pd.DataFrame(logs, columns=['Time', 'Direction', 'Source', 'Target']))

    n = st.slider("Choose the number of sentences", min_value=1, max_value=128)

    first_button, second_button, third_button = st.columns(3)
    with first_button:
        translate_button = st.button('Translate', type="primary", help="start translating!")
    with second_button:
        output_button = st.download_button("Output", key="output", file_name='logs.csv', data=logs_csv,
                                           mime='text/csv', help="output as the csv file.")
    with third_button:
        log_button = st.button('Log', type="secondary", help="translation logs~")

    input_texts = []
    input_ids = []
    output_containers = []
    output_texts = []

    for idx in range(n):
        input_text = st.text_input(f'Input  Sentence {idx + 1}:', key=idx)
        output_container = st.empty()
        if input_text:
            input_ids.append(idx)
            input_texts.append(input_text)

        output_containers.append(output_container)

    if translate_button:
        translate_click(msg,
                    input_texts,
                    input_ids,
                    target_language,
                    target_direction,
                    output_containers
                    )  
    if log_button:
        logs_click(msg)

### 子页面 文档翻译 ###

def second_body_content():
    st.title('File Translator')
    msg = st.empty()
    
    if st.button("Go to homePage", help="Please double click."):
        st.session_state.page = "homePage"
    
    upload_file = st.file_uploader(label="Please put in the txt, csv or json file")
    
    target_language = st.selectbox('Choose Target Language', ['English', 'Chinese'])
    
    match_translator_api = {
        'English': 'en',
        'Chinese': 'zh'
    }
    translate_directions = {
        'English': 'ZH → EN',
        'Chinese': 'EN → ZH'
    }

    target_direction = translate_directions[target_language]
    target_language = match_translator_api[target_language]
    
    input_texts = []
    input_ids = []
    output_containers = []
    output_texts = []
    
    first_button, second_button, third_button = st.columns(3)
    with first_button:
        translate_button = st.button('Translate', type="secondary")
    with second_button:
        logs_button = st.button('Logs', help="Check Logs.")
    
    my_bar = st.empty()

    if upload_file is not None:
        file_type = upload_file.name.split('.')[-1]
        block = st.empty()
        if file_type == "txt":
            txt_io = StringIO(upload_file.getvalue().decode("utf-8"))
            txt_data = txt_io.read().split('\n')
            data_show = pd.DataFrame(txt_data, columns=['Texts'])
            data_show.index += 1
            block.table(data_show)

            for idx in range(len(txt_data)):
                output_container = st.empty()
                output_containers.append(output_container)
                
                input_texts.append(txt_data[idx])
                input_ids.append(idx)
                
            
        elif file_type == "csv":
            data_show = pd.read_csv(upload_file,index_col=0)
            data_show = data_show
            block.table(data_show)
            
            df_list_texts = [data_show.columns.tolist()] + data_show.values.tolist()
            
            idx3 = 0
            for idx1 in range(len(df_list_texts)):
                for idx2 in range(len(df_list_texts[idx1])):
                    output_container = st.empty()
                    output_containers.append(output_container)
                    output_texts.append("TBD")
                    
                    input_texts.append(str(df_list_texts[idx1][idx2]))
                    input_ids.append(idx3)
                    idx3 += 1
            
        elif file_type == "json":
            json_data = json.load(upload_file)
            data_show = pd.DataFrame(json_data)
            block.table(data_show)
            
            df_list_texts = [data_show.keys().tolist()] + data_show.values.tolist()
            
            idx3 = 0
            for idx1 in range(len(df_list_texts)):
                for idx2 in range(len(df_list_texts[idx1])):
                    output_container = st.empty()
                    output_containers.append(output_container)
                    output_texts.append("TBD"+f"{idx3}")
                    
                    input_texts.append(str(df_list_texts[idx1][idx2]))
                    input_ids.append(idx3)
                    idx3 += 1
            

    
        if translate_button:
            if authorize():
                num_input = len(input_texts)
                if num_input == 0:
                    msg.error("Please put in the sentence you need to translate.", icon="🚨")
                else:
                    
                    not_cached = []
                    cached = []
                    for idx in range(num_input):
                        cached_text = cache.get((target_language, input_texts[idx]))
                        if cached_text is None:
                            not_cached.append(input_texts[idx])
                        else:
                            cached.append((idx, cached_text))

                    start_time = time.time()

                    input_time = time.asctime()

                    progress_bar = st.progress(0.3, text="Translation in progress…")
                
                    output_texts = translator.translate(not_cached, target_language)
                
                    end_time = time.time()

                    runtime = end_time - start_time
                    while True:
                        elapsed_time  = time.time()  - start_time
                        progress_bar.progress(min(elapsed_time / runtime, 1.0), text="Translation in progress…")
                        if elapsed_time >= runtime:
                            progress_bar.progress(min(elapsed_time / runtime, 1.0), text="Translation Complete!")
                            break

                    for idx in range(len(not_cached)):
                        cache[(target_language, not_cached[idx])] = output_texts[idx]

                    for idx, cached_text in cached:
                        output_texts.insert(idx, cached_text)

                    for idx in range(num_input):
                        logs.append([input_time, target_direction, input_texts[idx], output_texts[idx]])

                    msg.success("Runtime:{:.4f}sec Seq Num:{} Avg:{:.4f}sec/seq Cache hits:{}"
                                .format(runtime, num_input, runtime / num_input, len(cached)),
                                icon="🎈")   

                    if file_type == "txt":
                        data_show['Result'] = output_texts
                        block.table(data_show)
                    elif file_type == "json":
                        data_show[output_texts[0]] = pd.Series(output_texts[1:])
                        block.table(data_show)
                    elif file_type == "csv":
                        data_show[output_texts[0]] = pd.Series(output_texts)
                        block.table(data_show)
                    
            else:
                msg.error("Please input your Access Key", icon="🚨")   
                
        if logs_button:
            logs_click(msg)

    with third_button:
        if upload_file is not None:
            file_type = upload_file.name.split('.')[-1]
        
            if file_type == "txt":
                txt_download = ''''''
                
                for idx in range(len(output_texts)):
                    txt_download += '''''' + output_texts[idx] + '''\n'''
                st.download_button('Output', txt_download)
            
            elif file_type == "csv": 
                rows, columns = data_show.shape
                reshaped_output = pd.np.array(output_texts).reshape(rows+1, 1)
                csv_download = pd.DataFrame(reshaped_output[1:], columns=reshaped_output[0])
                csv_download.index += 1
                csv_download = codecs.BOM_UTF8 + csv_download.to_csv().encode('utf-8')
                st.download_button("Output", key="output", file_name='logs.csv', data=csv_download,
                                mime='text/csv', help="Click to download logs.")
                
            elif file_type == "json":
                rows, columns = data_show.shape   
                reshaped_output = pd.np.array(output_texts).reshape(rows+1, 1)
                json_download = pd.DataFrame(reshaped_output[1:], columns=reshaped_output[0]).to_json(orient='records', force_ascii=False, indent=2)
                st.download_button("Output", key="output", file_name='logs.json', data=json_download,
                                    help="Click to download logs.")
    
            


def home_page():
    st.session_state.page = "homePage"
    
    sidebar_widgets()
    home_body_content()

def second_page():
    st.session_state.page = "secondPage"
    
    sidebar_widgets()
    second_body_content()

def home():
    if "page" not in st.session_state:
        st.session_state.page = "homePage"
    
    if st.session_state.page == "homePage":
        home_page()
    elif st.session_state.page == "secondPage":
        second_page()

if __name__ == "__main__":
    home()

# 跨文化交流需要理解和尊重文化差异。
# 医疗保健部门不断适应医学研究和技术进展。
# 政府的角色是提供公共服务并确保公民的福祉。
# 全球化使经济和文化相互联系，塑造了一个更加紧密相连的世界。
# 音乐的力量在于它能够唤起情感并创造一种团结感。
# 环境保护对于保护生物多样性和减轻气候变化的影响至关重要。
# 教育体系在为年轻一代准备未来方面发挥着重要作用。
# 伦理考虑在各个领域的决策和保持诚信中至关重要。
# 量子力学对理论物理领域的影响是深远的，挑战了我们对现实的传统理解。


# The novel I'm currently reading is quite intriguing, with its complex characters and unexpected plot twists.
# In order to succeed, one must have a strong sense of determination, perseverance, and the ability to overcome obstacles.
# The scientific community is constantly striving to make new discoveries and push the boundaries of human knowledge.
# As technology continues to advance at a rapid pace, it is important for us to adapt and embrace these changes.
# Traveling allows us to broaden our horizons, experience different cultures, and gain a deeper understanding of the world.
# Effective communication is key in any relationship, as it fosters understanding, trust, and meaningful connections.
# Learning a new language requires dedication, practice, and a willingness to make mistakes and learn from them.
# The global economy is interconnected, and events in one country can have far-reaching impacts on others.
