import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, afterEach } from 'bun:test';

import Row from './Row';

describe('Row', () => {
  afterEach(cleanup);
  it('calculates the conversion correctly on input change', async () => {
    render(<Row scale={1} />);
    const input = screen.getByRole('spinbutton');

    await userEvent.type(input, '2');

    const result = await screen.findByText(/232.0 g/i);
    expect(result).not.toBeNull();
  });

  it('re-calculates when the unit is changed', async () => {
    render(<Row scale={1} />);
    const input = screen.getByRole('spinbutton');
    await userEvent.type(input, '1');

    // The initial result should be 116.0 g (1 * 116.0)
    expect(await screen.findByText(/116.0 g/i)).not.toBeNull();

    // Change the unit to Ounce (value: 8)
    const unitSelect = screen.getAllByRole('combobox')[0];
    await userEvent.click(unitSelect);
    await userEvent.click(await screen.findByText('Ounce'));

    // The new result should be 14.5 g (116.0 / 8)
    expect(await screen.findByText(/14.5 g/i)).not.toBeNull();
  });

  it('re-calculates when the ingredient is changed', async () => {
    render(<Row scale={1} />);
    const input = screen.getByRole('spinbutton');
    await userEvent.type(input, '1');

    // The initial result should be 116.0 g ('00' Pizza Flour)
    expect(await screen.findByText(/116.0 g/i)).not.toBeNull();

    // Change the ingredient to Agave syrup (value: 336.0)
    const ingredientSelect = screen.getAllByRole('combobox')[1];
    await userEvent.click(ingredientSelect);
    await userEvent.click(await screen.findByText('Agave syrup'));

    // The new result should be 336.0 g
    expect(await screen.findByText(/336.0 g/i)).not.toBeNull();
  });

  it('applies the scale prop correctly', async () => {
    render(<Row scale={2} />);
    const input = screen.getByRole('spinbutton');
    await userEvent.type(input, '1');

    // The result should be 232.0 g (1 * 116.0 * 2)
    expect(await screen.findByText(/232.0 g/i)).not.toBeNull();
  });
});
