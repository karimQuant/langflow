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
        self.browser_config = BrowserContextConfig(
            cookies_file=script_dir + "/cookies.json",
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
