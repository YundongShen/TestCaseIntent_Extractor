import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.when;

import java.util.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import domain.Estadistiques;
import domain.Partida;
import domain.Codi;
import domain.Codi.Colors;


@RunWith(MockitoJUnitRunner.class)
public class EstadistiquesTest {
    @Mock
    Partida p1;
    @Mock
    Partida p2;
    @Mock
    Partida p3;
   
    @Test
    public void testAfegirPartida(){
        assertNotNull(p1);
        when(p1.getTornActual()).thenReturn(2);
        when(p1.getCodi()).thenReturn(List.of(Colors.VERD,Colors.VERD,Colors.VERD,Colors.VERD));
        when(p1.partidaAcabada()).thenReturn(true);

        assertNotNull(p2);
        when(p2.getTornActual()).thenReturn(3);
        when(p2.getCodi()).thenReturn(List.of(Colors.VERD,Colors.VERD,Colors.VERD,Colors.VERD));
        when(p2.partidaAcabada()).thenReturn(true);

        assertNotNull(p3);
        when(p3.getCodi()).thenReturn(List.of(Colors.VERD,Colors.VERD,Colors.VERD,Colors.VERD,Colors.VERD));
        when(p3.partidaAcabada()).thenReturn(true);
        
        List<Partida> solucio = new ArrayList<Partida>();
        solucio.add(p3);
        solucio.add(p1);
        solucio.add(p2);

        Estadistiques est = new Estadistiques();
        est.afegirPartida(p1);
        est.afegirPartida(p2);
        est.afegirPartida(p3);

        assertEquals(est.getRanking(),solucio);
    }
}
