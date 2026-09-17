import unittest
from unittest.mock import MagicMock, patch, call
from unittest import IsolatedAsyncioTestCase
import asyncio
import logging
from src.token_operation import get_erc20_balance_behaviour
from src.handlers import crypto_filter_handler
from src.autonomous_agent import AutonomousAgent

# Configure logging to suppress asyncio debug logs
logging.basicConfig(level=logging.INFO)
asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.setLevel(logging.WARNING)  # Suppress asyncio debug logs


class TestAutonomousAgent(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()

        self.agent = AutonomousAgent("TestAgent")
        self.agent.inbox = self.inbox
        self.agent.outbox = self.outbox
        self.processing_task = asyncio.create_task(self.agent.process_messages())

    async def asyncTearDown(self):
        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass

    @patch("src.handlers.logger.info")
    async def test_hello_handler(self, mock_logger_info):
        print("Executing test_hello_handler...")
        message = {"type": "hello", "content": "hello world"}
        await self.inbox.put(message)
        await asyncio.sleep(1)
        mock_logger_info.assert_any_call("TestAgent:: Filtered message (hello)")

    @patch("src.handlers.logger.info")
    @patch("src.token_operation.transfer_erc20_token")
    async def test_crypto_handler(self, mock_transfer_erc20_token, mock_logger_info):
        mock_transaction_hash = "0x9af63e625bd695c922cba4fdf1cd3475974ed0b636d4d518096701b9f90f012a"
        mock_transfer_erc20_token.return_value = mock_transaction_hash

        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.w3 = MagicMock()
        mock_agent.w3.eth.get_balance.return_value = 2000000000000000000  # 2 ETH
        mock_agent.w3.eth.send_raw_transaction.return_value = MagicMock(
            hex=MagicMock(return_value=mock_transaction_hash)
        )

        mock_agent.token_contract = MagicMock()
        mock_agent.token_contract.functions.decimals.return_value.call.return_value = 18
        mock_agent.token_contract.functions.balanceOf.return_value.call.return_value = 1000000000000000000
        mock_agent.token_decimals = 18

        message = {"type": "crypto", "content": "crypto transfer"}
        await self.inbox.put(message)

        await crypto_filter_handler(mock_agent, message)

        mock_logger_info.assert_has_calls([
            call("TestAgent:: Initiating ERC-20 token transfer..."),
            call(f"TestAgent:: Transaction sent with hash: {mock_transaction_hash}")
        ])


class TestERC20BalanceBehaviour(IsolatedAsyncioTestCase):
    @patch("src.autonomous_agent.Web3")
    @patch("src.token_operation.logger.info")
    async def test_get_erc20_balance_behaviour(self, mock_logger_info, mock_web3):
        with patch.dict("src.config.settings.ETH_SETTINGS", {
            "FROM_ADDRESS": "0xMockedAddress",
            "TOKEN_ADDRESS": "0xMockedTokenAddress",
            "RPC_NODE_URL": "https://mocked.node.url",
        }):
            mock_web3.return_value.is_connected.return_value = True

            mock_token_contract = MagicMock()
            mock_token_contract.functions.balanceOf.return_value.call.return_value = 1000000000000000000
            mock_token_contract.functions.decimals.return_value.call.return_value = 18
            mock_web3.return_value.eth.contract.return_value = mock_token_contract

            mock_agent = MagicMock(spec=AutonomousAgent)
            mock_agent.name = "TestAgent"
            mock_agent.token_contract = mock_token_contract
            mock_agent.token_decimals = 18

            with patch("asyncio.sleep", side_effect=asyncio.CancelledError):
                try:
                    await get_erc20_balance_behaviour(mock_agent)
                except asyncio.CancelledError:
                    pass

            mock_token_contract.functions.balanceOf.assert_called_once_with("0xMockedAddress")
            mock_logger_info.assert_called_once_with("TestAgent:: ERC-20 Token Balance for address 0xMockedAddress: 1.0")


if __name__ == "__main__":
    unittest.main()
