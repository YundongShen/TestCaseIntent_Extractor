import React from "react";
import { expect, it } from "vitest";
import { render, screen } from '@testing-library/react';
import App from './src/App';
import userEvent from "@testing-library/user-event";

// No need for `act` when using `userEvent` for interactions
it('adds and removes items correctly', async () => {
  const user = userEvent.setup();
  render(<App/>);

  const getItems = () => screen.queryAllByRole('listitem');
  const noItemsMessage = () => screen.queryByText(/no items yet/i);

  // Initial State: No items are visible
  expect(getItems()).toHaveLength(0);
  expect(noItemsMessage()).toBeVisible();

  // Click the "Add Item" button
  const btn = screen.getByRole('button', {name: /add item/i});
  await user.click(btn);

  // After Clicking: One item is visible and the message is gone
  expect(getItems()).toHaveLength(1);
  expect(getItems()[0].textContent).toEqual('Item 1');
  expect(noItemsMessage()).not.toBeInTheDocument();
});
