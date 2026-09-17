Feature: US06 - Alertas Automatizadas sobre condiciones críticas

  Scenario: Alerta por enfermedad detectada
    Given que el sistema detecta una enfermedad potencial, 
    When el nivel de riesgo supera el umbral,
    Then se envía una alerta al celular del usuario.

    Examples:

    | Datos de los sensoers e imagenes del cultivo |
    | Detección de estado anormal en el cultivo | Busqueda de enfermedad o plaga en el cultivo |
    | Enfermedad o plaga especifica detectada | Notificar al usuario sobre la plaga o enfermedad |

    | Alerta de notificacion al Usuario: |
    | Plaga/Enfermedad detectada en sus cultivos, revisar datos de la plaga/enfermedad |
    
  Scenario: Alerta por evento climático
    Given que se pronostica una helada, 
    When el evento se acerca, 
    Then se envía una notificación para prevenir daños.

    Examples:
    
    | Pronostico climatico con evento extremo resivido |
    | Notificar al usuario sobre el pronostico climaco con un evento extremo |

    | Alerta de notificacion al Usuario: |
    | Pronostico climatico extremo en la zona de cultivo, revise las medidas preventivas para perdias minimas en tus cultivos |    
