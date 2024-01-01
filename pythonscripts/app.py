# Python libraries
import streamlit as sl
import logging

from chatbotModel import LlmChatbot
from config import ConfigUI, ConfigLog


def initialize_chatbot():
    chatbot_exists = 'chatbot' in sl.session_state
    chatbot_history_exists = chatbot_exists and sl.session_state.chatbot.history

    if not chatbot_history_exists:
        logging.info(f"Chatbot history: False")
        sl.session_state.chatbot = LlmChatbot(history = False)
    else:
        logging.info(f"Chatbot history: True")


def display_results(result):
    sl.write(f'{ConfigUI.BOT_ICON} ', result.get('result', 'No answer found'))

    source_documents = result.get('source_documents', [])
    if source_documents:
        sl.write(f'{ConfigUI.HISTORY_ICON} ', source_documents)


def collect_user_feedback():
    user_feedback = sl.text_input('Provide feedback (optional): ')
    if user_feedback:
        sl.write('Thank you for your feedback! We value your input.')


def create_history_sidebar():
    with sl.sidebar:
        sl.subheader('Question History')

        for entry in sl.session_state.chatbot.history:
            sl.write(f"Q: {entry['question']}")
            sl.write(f"A: {entry['answer'].get('result', 'No answer found')}")
            sl.write("---")


def user_QA():
    with sl.form('qa_form'):
        user_input = sl.text_input('Enter your question: ')
        submit_button = sl.form_submit_button('Send')

        if submit_button:
            # Retrieve the conversation history
            history = [entry['question'] for entry in sl.session_state.chatbot.history]

            logging.debug(f"User input: {user_input}")
            logging.debug(f"Existing history: {sl.session_state.chatbot.history}")

            with sl.spinner('Searching for an answer...'):
                # Pass the conversation history to the final_result method
                result = sl.session_state.chatbot.final_result(user_input, conversation_history = history)

            logging.debug(f"Result: {result}")
            display_results(result)
            collect_user_feedback()

            sl.session_state.chatbot.history.append({'question': user_input, 'answer': result})


def chatbotUI():
    logging.info("Starting the Chatbot!")

    sl.set_page_config(page_title = ConfigUI.PAGE_TITLE,
                       page_icon = ConfigUI.PAGE_ICON,
                       layout = ConfigUI.PAGE_LAYOUT)
    sl.header(f'Personalized Chatbot {ConfigUI.TITLE_ICON}')

    initialize_chatbot()
    create_history_sidebar()
    user_QA()


def configure_logging():
    logging.basicConfig(format = ConfigLog.LOG_FORMAT, level = logging.INFO)


if __name__ == "__main__":
    configure_logging()
    chatbotUI()
