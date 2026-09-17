import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import domain.Estadistiques;
import domain.Partida;

@RunWith(MockitoJUnitRunner.class)
public class EstadistiquesTest {
    @Mock
    Partida p1;
    @Mock
    Partida p2;
    @Mock
    Partida p3;
    @Mock
    Partida p4;

    @Test
    public void testAfegirPartida() {
        assertNotNull(p1);
        when(p1.getRondes()).thenReturn(2);
        when(p1.getPuntuacioHuma()).thenReturn(4);
        when(p1.getPuntuacioMaquina()).thenReturn(12);
        when(p1.partidaAcabada()).thenReturn(true);
        when(p1.posicions()).thenReturn(4);

        assertNotNull(p2);
        when(p2.getPuntuacioHuma()).thenReturn(8);
        when(p2.getPuntuacioMaquina()).thenReturn(12);
        when(p2.partidaAcabada()).thenReturn(true);
        when(p2.posicions()).thenReturn(4);

        assertNotNull(p3);
        when(p3.getRondes()).thenReturn(4);
        when(p3.getPuntuacioHuma()).thenReturn(4);
        when(p3.getPuntuacioMaquina()).thenReturn(12);
        when(p3.partidaAcabada()).thenReturn(true);
        when(p3.posicions()).thenReturn(4);

        assertNotNull(p4);
        when(p4.getPuntuacioHuma()).thenReturn(4);
        when(p4.getPuntuacioMaquina()).thenReturn(12);
        when(p4.partidaAcabada()).thenReturn(true);
        when(p4.posicions()).thenReturn(6);

        List<Partida> solucio = new ArrayList<Partida>();
        solucio.add(p4);
        solucio.add(p2);
        solucio.add(p3);
        solucio.add(p1);

        System.out.println(solucio);

        Estadistiques est = new Estadistiques();
        est.afegirPartida(p1);
        est.afegirPartida(p2);
        est.afegirPartida(p3);
        est.afegirPartida(p4);

        System.out.println(est.getRanking());

        assertEquals(est.getRanking(), solucio);
    }
}
