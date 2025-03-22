import pandas as pd
import os
import json
from typing import Dict, Any, Optional, List
from langflow.schema import Data

from langflow.custom import Component
from langflow.inputs import MessageTextInput, IntInput, DropdownInput
from langflow.template import Output

from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, Controller
from dotenv import load_dotenv
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class BrowserUse(Component):
    display_name = "Browser Use Component"
    description = "An agent that can interact with a browser to perform tasks."
    inputs = [
        MessageTextInput(
            name="task",
            display_name="Task",
            info="The specific task instruction for the agent",
            value="Find trending tokens on Solana and Ethereum chains",
            required=True,
        ),
        MessageTextInput(
            name="browser_init_action",
            display_name="Initial Action",
            info="a dict of initial actions https://github.com/browser-use/browser-use/blob/main/examples/features/initial_actions.py",
            value="""[{"open_tab": {"url": "https://dexscreener.com/"}}]""",
            required=False,
        )
    ]
    outputs = [
        Output(display_name="Data", name="output_data", method="build_output"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        script_dir=os.path.dirname(os.path.abspath(__file__))
        self.browser_llm = ChatOpenAI(model="gpt-4o")
        self.browser = Browser()
        self.brower_controller = Controller()
        cookies_path = script_dir + "/browser_cookies.json"
        logger.info(f"loading cookies from {cookies_path}")
        if not os.path.exists(cookies_path):
            raise FileNotFoundError(
                f"Browser cookies file not found at {cookies_path}"
            )
        self.browser_config = BrowserContextConfig(
            cookies_file=script_dir + "/browser_cookies.json",
            wait_for_network_idle_page_load_time=3.0,
            browser_window_size={"width": 1280, "height": 1100},
            locale="en-US",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            highlight_elements=False,
            viewport_expansion=500,
        )
        self.browser_context = BrowserContext(
            browser=self.browser, config=self.browser_config
        )

    async def build_output(self) -> Data:
        try:
            initial_actions = json.loads(self.browser_init_action)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format for initial actions. Using empty list.")
            initial_actions = []
        agent = Agent(
            llm=self.browser_llm,
            initial_actions=initial_actions,
            task=self.task,
            browser_context=self.browser_context,
            controller=self.brower_controller,
        )
        history = await agent.run()
        tokens = json.loads(history.final_result())
        # tokens=Tokens(tokens=[])
        return Data(tokens=tokens)

if __name__ == "__main__":
    browser_use = BrowserUse()
    input_data = {
        "task": "Find trending tokens on Solana and Ethereum chains",
        "browser_init_action": '[{"open_tab": {"url": "https://dexscreener.com/"}}]',
    }
    output_data = browser_use.run(input_data)