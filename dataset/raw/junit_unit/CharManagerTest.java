package ua.foxminded.javaspring.CharManagerTest;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import ua.foxminded.javaspring.CharCalculator.CharCalculator;
import ua.foxminded.javaspring.CharManager.CharManager;
import ua.foxminded.javaspring.charCounter.CharCounter;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

public class CharManagerTest {
    @Mock
    private CharCalculator calculatorMock;

    @Mock
    private CharCounter cacheMock;

    @InjectMocks
    private CharManager manager;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        manager = new CharManager();
        manager.setCalculator(calculatorMock);
        manager.setCache(cacheMock);
    }

    @Test
    public void countUniqueChars_shouldReturnCachedResult_whenCacheHit() {
        String inputString = "Hello, World!";
        String expected = "'H' - 1\n' ' - 1\n'W' - 1\n'o' - 2\n'r' - 1\n'l' - 3\n',' - 1\n'd' - 1\n'!' - 1\n'e' - 1\n";

        // Cache hit
        when(cacheMock.get(inputString)).thenReturn(expected);
        String result = manager.countUniqueChars(inputString);

        verify(calculatorMock, never()).countUniqueChars(any());
        assertEquals(expected, result);
    }

    @Test
    public void countUniqueChars_shouldCalculateResultAndCache_whenCacheMiss() {
        String inputString = "Hello, World!";
        String expected = "'H' - 1\n' ' - 1\n'W' - 1\n'o' - 2\n'r' - 1\n'l' - 3\n',' - 1\n'd' - 1\n'!' - 1\n'e' - 1\n";

        // Cache miss
        when(cacheMock.get(inputString)).thenReturn(null);
        when(calculatorMock.countUniqueChars(inputString)).thenReturn(expected);
        String result = manager.countUniqueChars(inputString);

        verify(calculatorMock, times(1)).countUniqueChars(inputString);
        assertEquals(expected, result);
    }

    @Test
    public void countUniqueChars_shouldReturnResultFromCalculator_whenNotCached() {
        // Arrange
        CharCalculator calculatorMock = mock(CharCalculator.class);
        CharCounter cacheMock = mock(CharCounter.class);
        CharManager manager = new CharManager();
        manager.setCalculator(calculatorMock);
        manager.setCache(cacheMock);

        String inputString = "Hello, World!";
        String expected = "'H' - 1\n' ' - 1\n'e' - 1\n'l' - 3\n'o' - 2\n',' - 1\n'W' - 1\n'r' - 1\n'd' - 1\n'!' - 1";

        // Simulate cache miss
        when(cacheMock.get(inputString)).thenReturn(null);
        when(calculatorMock.countUniqueChars(inputString)).thenReturn(expected);

        // Act
        String result = manager.countUniqueChars(inputString);

        // Assert
        verify(cacheMock, times(1)).get(inputString);
        verify(cacheMock, times(1)).put(inputString, expected);
        verify(calculatorMock, times(1)).countUniqueChars(inputString);
        assertEquals(expected, result);
    }

    @Test
    public void countUniqueChars_shouldReturnCachedResult_whenAlreadyCached() {
        // Arrange
        CharCalculator calculatorMock = mock(CharCalculator.class);
        CharCounter cacheMock = mock(CharCounter.class);
        CharManager manager = new CharManager();
        manager.setCalculator(calculatorMock);
        manager.setCache(cacheMock);

        String inputString = "Hello, World!";
        String expected = "'H' - 1\n' ' - 1\n'e' - 1\n'l' - 3\n'o' - 2\n',' - 1\n'W' - 1\n'r' - 1\n'd' - 1\n'!' - 1";

        // Simulate cache hit
        when(cacheMock.get(inputString)).thenReturn(expected);

        // Act
        String result = manager.countUniqueChars(inputString);

        // Assert
        verify(cacheMock, times(1)).get(inputString);
        verify(cacheMock, never()).put(any(), any());
        verify(calculatorMock, never()).countUniqueChars(any());
        assertEquals(expected, result);
    }

    @Test
    public void countUniqueChars_shouldReturnEmptyString_whenInputIsEmpty() {
        // Arrange
        String inputString = "";
        String expected = "";

        // Cache hit
        when(cacheMock.get(inputString)).thenReturn(expected);

        // Act
        String result = manager.countUniqueChars(inputString);

        // Assert
        verify(calculatorMock, never()).countUniqueChars(any());
        assertEquals(expected, result);
    }
}
