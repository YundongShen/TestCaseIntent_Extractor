import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import com.ruoran.tosu.dao.TaskDaoImpl;
import com.ruoran.tosu.model.Task;

import static org.mockito.Mockito.*;


public class FirstTest {

    @Mock
    TaskDaoImpl mockDao; 
    
    @Mock
    Task task;
    
    @Before
    public void setup() {
        MockitoAnnotations.initMocks(this);
        
        when(mockDao.create(anyString(), anyInt())).thenReturn("a_key");
        when(mockDao.get("a_key")).thenReturn(task);
    }
    
    @Test
    public void testCreate() {
        // first test
        String key = mockDao.create("hi", 12);
        assertEquals(key, "a_key");
    }
    
    @Test
    public void testGet() {
        // first test
        Task a = mockDao.get("a_key");
        assertEquals(a, task);
    }

}