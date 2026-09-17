Feature: US13 - Accesibilidad y navegación inicial

  Scenario: Navegador de diferentes cultivos
    Given que el usuario quiera ver detalles de una planta en específico, 
    When entre al navegar de cultivos y escriba el nombre de la planta, 
    Then le aparecerá una lista de plantas cuyo nombre coincide.

    Examples:
    
    | Planta buscada en el navegador|Filtro: plantas|Buscar|

    | Foto basica de planta 1 | Nombre de planta 1 | Datos basicos de de planta 1 |
    | Foto basica de planta 2 | Nombre de planta 2 | Datos basicos de de planta 2 |
    | Foto basica de planta 3 | Nombre de planta 3 | Datos basicos de de planta 3 |
    | Foto basica de planta 4 | Nombre de planta 4 | Datos basicos de de planta 4 |

  Scenario: Ver detalles de una planta
    Given que el usuario encuentre la planta buscada en el navegador, 
    When haga click en la planta deseada,
    Then le abrirá una página detallada sobre las necesidades y propiedades de esta planta.

    Examples:

    | Nombre de la planta seleccionada |
    | Foto de la plaanta seleccionada |
    | Datos basicos |
    | Necesidades de la planta |
    | Datos especificos de la planta |
