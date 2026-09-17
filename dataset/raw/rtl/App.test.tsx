import { expect, test, describe, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { render as rtlRender, RenderResult, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { weatherMock, geoLocationMock, mockNewYorkWeather } from '../tests/mocks';
import App from './App';
import { UnitProvider } from './libs/hooks/use-unit';

const handlers = [
  http.get('https://api.openweathermap.org/geo/1.0/direct', () => {
    return HttpResponse.json(geoLocationMock);
  }),
  http.get('https://api.openweathermap.org/data/3.0/onecall', () => {
    return HttpResponse.json(weatherMock);
  }),
];

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function render<T>(_ui: React.ReactElement<T>): RenderResult {
  return {
    ...rtlRender(
      <QueryClientProvider client={queryClient}>
        <UnitProvider>{_ui}</UnitProvider>
      </QueryClientProvider>,
    ),
  };
}

describe('App', () => {
  test('it should render the app', async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText('Los Angeles')).toBeInTheDocument());
  });

  test('it should render the app with error', async () => {
    server.use(
      http.get('https://api.openweathermap.org/data/3.0/onecall', () => {
        return new HttpResponse('Error', {
          status: 500,
        });
      }),
    );
    render(<App />);

    await waitFor(() => expect(screen.getByText('An error occurred')).toBeInTheDocument());
  });

  // render with celsius
  test('it should render the app with celsius', async () => {
    localStorage.setItem('unit', 'Fahrenheit');
    render(<App />);

    await waitFor(() => expect(screen.getByText('Los Angeles')).toBeInTheDocument());

    const weatherStatus = screen.getByTestId('weather-status');
    expect(weatherStatus).toHaveTextContent('62° F');
  });

  // set new location
  test('it should set new location', async () => {
    render(<App />);

    server.use(
      http.get('https://api.openweathermap.org/geo/1.0/direct', () => {
        return HttpResponse.json([
          {
            name: 'New York County',
            lat: 40.7143,
            lon: -74.006,
            country: 'US',
            state: 'New York',
          },
        ]);
      }),
    );

    server.use(
      http.get('https://api.openweathermap.org/data/3.0/onecall', () => {
        return HttpResponse.json(mockNewYorkWeather);
      }),
    );

    await waitFor(() => expect(screen.getByText('Los Angeles')).toBeInTheDocument());

    userEvent.type(screen.getByTestId('search-input'), 'New Work');

    await waitFor(() => screen.getByText('New York County - New York - US'));

    userEvent.click(screen.getByText('New York County - New York - US'));

    await waitFor(() => expect(screen.getByText('New York County')).toBeInTheDocument());
  });
});
