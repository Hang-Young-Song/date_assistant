import streamlit as st
import pandas as pd
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# streamlit run "C:/Users/ssb70/OneDrive/바탕 화면/code8/code8/실습 코드/P08_CH04_date_assistant/P08_CH04_02_streamlit_date_assistant.py"
# streamlit window wide로

st.set_page_config(layout="wide")

st.header("💬 데이팅 어시스턴트")

USER_NAME = "나"
AI_NAME = "CuPT"

# OpenAI API 키 입력
api_key = st.text_input("OpenAI API Key", type="password")

def get_suggestion(messages, api_key, num_candi=3):

    conv = ""
    for message in messages[1:]:
        name = USER_NAME if message['role'] == 'user' else AI_NAME
        conv += f"{name}: {message['content']}"

    eval_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.8, openai_api_key=api_key)  # CoT는 다양한 샘플을 만들어야 하기 때문에 temperature를 올려야 함
    basic_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, openai_api_key=api_key)  # CoT는 다양한 샘플을 만들어야 하기 때문에 temperature를 올려야 함

    class Suggestion(BaseModel):
        sentiment: str = Field(description="대화의 분위기: Positive, Negative, Neutral")
        suggestion_text: str = Field(description="대화에 대한 markdown형식의 자세한 분석과 적절한 조언")

    parser = JsonOutputParser(pydantic_object=Suggestion)
    format_instructions = parser.get_format_instructions()

    human_prompt_template = HumanMessagePromptTemplate.from_template(
        "{conv}\n위 대화 내용은 대학생 소개팅 상황에서 처음 만나는 남녀의 대화이다. 위 대화를 분석하고 {name}에게 markdown형식으로 적절한 조언을 하라.\n{format_instructions}")

    suggestion_gen_prompt = ChatPromptTemplate.from_messages(
        [
            human_prompt_template,
        ])
    suggestion_gen_prompt = suggestion_gen_prompt.partial(format_instructions=format_instructions)

    suggestion_gen_chain = suggestion_gen_prompt | basic_model | parser

    class VoteCoT(BaseModel):
        thought: str = Field(description="voting number를 선택한 이유에 대해 자세히 넣어주세요.")
        voting_num: int = Field(description="voting number")

    parser = JsonOutputParser(pydantic_object=VoteCoT)
    format_instructions = parser.get_format_instructions()

    voting_prompt_template = HumanMessagePromptTemplate.from_template(
        "{conv}\n다음은 위 대학생 소개팅 대화에서 {name}에게 하면 좋을 조언이다. 아래의 후보 중 가장 좋은 것을 추론 과정과 함께 투표 번호를 응답하라.\n{candidates}\n{format_instructions}")

    voting_prompt = ChatPromptTemplate.from_messages(
        [
            voting_prompt_template,
        ])
    voting_prompt = voting_prompt.partial(format_instructions=format_instructions)
    voting_chain = voting_prompt | eval_model | parser

    suggestion_list = suggestion_gen_chain.batch([{"conv": conv, "name": USER_NAME}] * num_candi)

    yield "## Suggestion candidates\n"
    yield "\n---\n".join([f"- {i} th\n- {sug['sentiment']}\n- {sug['suggestion_text']}\n" for i, sug in enumerate(suggestion_list)])

    candidates = "\n\n".join([f"후보 {i}.\n{suggestion}" for i, suggestion in enumerate(suggestion_list)])
    vote_list = voting_chain.batch([{"conv": conv, "name": USER_NAME, "candidates": candidates}] * num_candi)

    df = pd.DataFrame(vote_list)
    yield "## Voting\n"
    yield df
    print(df)

    best_candi_num = df['voting_num'].mode()[0]
    best_suggestion = suggestion_list[best_candi_num]

    yield "## Best Suggestion\n"
    yield f"{best_candi_num} th\n\n{best_suggestion['sentiment']}\n\n{best_suggestion['suggestion_text']}"

if api_key:
    openai.api_key = api_key

    if "messages" not in st.session_state:
        system_prompt = f"""\
        너는 20대 여성, 전공은 통계학과이고 아래의 프로필을 따라 응답한다.
        - 이름: 수연
        - 나이: 23
        - '처음' 만나는 1:1 소개팅 상황이다. 커피집에서 만났다.
        - 소개팅이기에 너무 도움을 주려고 대화하지 않는다. 자연스러운 대화를한다.
        - 너무 적극적으로 이야기하지는 않는다.
        - 대화를 리드하지 않는다.
        - 수동적으로 대답한다.
        """
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    user_input = st.chat_input("What is up?")

    col1, col2 = st.columns(2)
    with col1:
        for message in st.session_state.messages[1:]:
            avatar = "🧑" if message['role'] == 'user' else "👩🏼"
            with st.chat_message(avatar):
                st.markdown(message["content"])

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("🧑"):
                st.markdown(user_input)

            with st.chat_message("👩🏼"):
                stream = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = ""
                for chunk in stream:
                    response += chunk['choices'][0]['delta'].get('content', '')
                    st.write(response, end="")

                st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        if len(st.session_state.messages) > 2:
            with st.spinner("분석중..."):
                stream = get_suggestion(st.session_state.messages, api_key)
                response = st.write_stream(stream)

else:
    st.warning("OpenAI API Key를 입력해주세요.")
