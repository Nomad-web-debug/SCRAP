-- Crear tabla de documentos
CREATE TABLE IF NOT EXISTS documentos (
    id SERIAL PRIMARY KEY,
    nombre_archivo VARCHAR(255) NOT NULL,
    fecha_documento DATE,
    fecha_procesamiento TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(nombre_archivo)
);

-- Crear tabla de contenido clasificado
CREATE TABLE IF NOT EXISTS contenido_clasificado (
    id SERIAL PRIMARY KEY,
    documento_id INTEGER REFERENCES documentos(id),
    categoria VARCHAR(50) NOT NULL,
    contenido JSONB NOT NULL,
    metricas JSONB,
    fecha_creacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(documento_id, categoria)
);

-- Índices para mejorar el rendimiento
CREATE INDEX IF NOT EXISTS idx_documentos_fecha ON documentos(fecha_documento);
CREATE INDEX IF NOT EXISTS idx_contenido_categoria ON contenido_clasificado(categoria);
CREATE INDEX IF NOT EXISTS idx_contenido_documento ON contenido_clasificado(documento_id);

-- Crear tabla de normas legales
CREATE TABLE IF NOT EXISTS normas_legales (
    id VARCHAR(50) PRIMARY KEY,
    categoria_principal VARCHAR(100),
    subcategoria_1 VARCHAR(100),
    subcategoria_2 VARCHAR(100),
    subcategoria_3 VARCHAR(100),
    titulo_numero VARCHAR(10),      -- Nuevo: número del título (ej: "TÍTULO I")
    titulo_nombre TEXT,            -- Nuevo: nombre del título
    capitulo_numero VARCHAR(10),    -- Nuevo: número del capítulo (ej: "CAPÍTULO II")
    capitulo_nombre TEXT,          -- Nuevo: nombre del capítulo
    seccion_numero VARCHAR(10),     -- Nuevo: número de sección si existe
    seccion_nombre TEXT,           -- Nuevo: nombre de la sección
    articulo VARCHAR(50),
    titulo TEXT,
    texto_norma TEXT NOT NULL,
    palabras_clave JSONB,
    fuente_url VARCHAR(255),
    origen VARCHAR(100),
    nombre_archivo VARCHAR(255),
    fecha_procesamiento TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    fecha_extraccion DATE NOT NULL,
    estado_vigencia VARCHAR(50),
    tipo_norma VARCHAR(100),
    numero_norma VARCHAR(100),
    entidad_emisora VARCHAR(200),
    ambito_aplicacion VARCHAR(100),
    referencias_normativas JSONB,
    modificaciones JSONB,
    observaciones TEXT
);

-- Índices para mejorar el rendimiento
CREATE INDEX IF NOT EXISTS idx_normas_categoria ON normas_legales(categoria_principal);
CREATE INDEX IF NOT EXISTS idx_normas_subcategoria1 ON normas_legales(subcategoria_1);
CREATE INDEX IF NOT EXISTS idx_normas_fecha_extraccion ON normas_legales(fecha_extraccion);
CREATE INDEX IF NOT EXISTS idx_normas_estado_vigencia ON normas_legales(estado_vigencia);
CREATE INDEX IF NOT EXISTS idx_normas_tipo ON normas_legales(tipo_norma);
CREATE INDEX IF NOT EXISTS idx_normas_palabras_clave ON normas_legales USING GIN (palabras_clave);
CREATE INDEX IF NOT EXISTS idx_normas_texto ON normas_legales USING GIN (to_tsvector('spanish', texto_norma));
CREATE INDEX IF NOT EXISTS idx_normas_referencias ON normas_legales USING GIN (referencias_normativas);

-- Nuevos índices para la estructura jerárquica
CREATE INDEX IF NOT EXISTS idx_normas_titulo_num ON normas_legales(titulo_numero);
CREATE INDEX IF NOT EXISTS idx_normas_capitulo_num ON normas_legales(capitulo_numero);
CREATE INDEX IF NOT EXISTS idx_normas_seccion_num ON normas_legales(seccion_numero);

-- Función para búsqueda de texto completo
CREATE OR REPLACE FUNCTION buscar_normas(query text) 
RETURNS TABLE (
    id VARCHAR(50),
    titulo TEXT,
    categoria_principal VARCHAR(100),
    tipo_norma VARCHAR(100),
    estado_vigencia VARCHAR(50),
    fecha_extraccion DATE,
    score float4
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        n.id,
        n.titulo,
        n.categoria_principal,
        n.tipo_norma,
        n.estado_vigencia,
        n.fecha_extraccion,
        ts_rank_cd(to_tsvector('spanish', n.texto_norma), plainto_tsquery('spanish', query)) as score
    FROM normas_legales n
    WHERE to_tsvector('spanish', n.texto_norma) @@ plainto_tsquery('spanish', query)
    ORDER BY score DESC;
END;
$$ LANGUAGE plpgsql;

-- Crear tabla de artículos
CREATE TABLE IF NOT EXISTS articulos (
    id SERIAL PRIMARY KEY,
    documento_id VARCHAR(50) REFERENCES normas_legales(id),
    numero VARCHAR(10) NOT NULL,
    contenido TEXT NOT NULL,
    fecha_creacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(documento_id, numero)
);

-- Crear tabla para errores de validación
CREATE TABLE IF NOT EXISTS errores_validacion (
    id SERIAL PRIMARY KEY,
    documento_id VARCHAR(50) REFERENCES normas_legales(id),
    tipo_error VARCHAR(50) NOT NULL,
    descripcion TEXT NOT NULL,
    fecha_creacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Índices para las nuevas tablas
CREATE INDEX IF NOT EXISTS idx_articulos_doc ON articulos(documento_id);
CREATE INDEX IF NOT EXISTS idx_articulos_num ON articulos(numero);
CREATE INDEX IF NOT EXISTS idx_errores_doc ON errores_validacion(documento_id); 