
#include "PyOptixDecls.h"
#include <numpy/arrayobject.h>


static void getNumpyElementType( RTformat format, int* element_py_array_type, unsigned int* element_dimensionality )
{
  switch( format )
  {
    case RT_FORMAT_HALF:
    case RT_FORMAT_HALF2:
    case RT_FORMAT_HALF3:
    case RT_FORMAT_HALF4:
      *element_py_array_type  = NPY_FLOAT16;
      *element_dimensionality = 1 + format - RT_FORMAT_HALF;
      return;

    case RT_FORMAT_FLOAT:
    case RT_FORMAT_FLOAT2:
    case RT_FORMAT_FLOAT3:
    case RT_FORMAT_FLOAT4:
      *element_py_array_type  = NPY_FLOAT32;
      *element_dimensionality = 1 + format - RT_FORMAT_FLOAT;
      return;

    case RT_FORMAT_BYTE:
    case RT_FORMAT_BYTE2:
    case RT_FORMAT_BYTE3:
    case RT_FORMAT_BYTE4:
      *element_py_array_type  = NPY_INT8;
      *element_dimensionality = 1 + format - RT_FORMAT_BYTE;
      return;

    case RT_FORMAT_UNSIGNED_BYTE:
    case RT_FORMAT_UNSIGNED_BYTE2:
    case RT_FORMAT_UNSIGNED_BYTE3:
    case RT_FORMAT_UNSIGNED_BYTE4:
      *element_py_array_type  = NPY_UINT8;
      *element_dimensionality = 1 + format - RT_FORMAT_UNSIGNED_BYTE;
      fprintf( stderr, "1 + %i - %i = %i\n", format, RT_FORMAT_UNSIGNED_BYTE, 1 + format - RT_FORMAT_UNSIGNED_BYTE );  
      return;

    case RT_FORMAT_SHORT:
    case RT_FORMAT_SHORT2:
    case RT_FORMAT_SHORT3:
    case RT_FORMAT_SHORT4:
      *element_py_array_type  = NPY_INT16;
      *element_dimensionality = 1 + format - RT_FORMAT_SHORT;
      return;

    case RT_FORMAT_UNSIGNED_SHORT:
    case RT_FORMAT_UNSIGNED_SHORT2:
    case RT_FORMAT_UNSIGNED_SHORT3:
    case RT_FORMAT_UNSIGNED_SHORT4:
      *element_py_array_type  = NPY_UINT16;
      *element_dimensionality = 1 + format - RT_FORMAT_UNSIGNED_SHORT;
      return;

    case RT_FORMAT_INT:
    case RT_FORMAT_INT2:
    case RT_FORMAT_INT3:
    case RT_FORMAT_INT4:
      *element_py_array_type  = NPY_INT32;
      *element_dimensionality = 1 + format - RT_FORMAT_INT;
      return;

    case RT_FORMAT_UNSIGNED_INT:
    case RT_FORMAT_UNSIGNED_INT2:
    case RT_FORMAT_UNSIGNED_INT3:
    case RT_FORMAT_UNSIGNED_INT4:
      *element_py_array_type  = NPY_UINT32;
      *element_dimensionality = 1 + format - RT_FORMAT_UNSIGNED_INT;
      return;

    case RT_FORMAT_BUFFER_ID:
    case RT_FORMAT_PROGRAM_ID:
      *element_py_array_type  = NPY_INT32;
      *element_dimensionality = 1;
      return;

    case RT_FORMAT_USER:
    case RT_FORMAT_UNKNOWN:
      *element_py_array_type  = 0;
      *element_dimensionality = 0;
      return;

  }
}

static PyObject* createNumpyArray( RTbuffer buffer, void* data )
{

  unsigned int dimensionality;
  rtBufferGetDimensionality( buffer, &dimensionality );

  RTsize rt_dims[3] = {0};
  rtBufferGetSizev( buffer, dimensionality, rt_dims );

  RTformat format;
  rtBufferGetFormat( buffer, &format );

  int element_py_array_type;
  unsigned int element_dimensionality;
  getNumpyElementType( format, &element_py_array_type, &element_dimensionality );

  fprintf( stderr, "format: %i %i\n", format, RT_FORMAT_UNSIGNED_BYTE );
  fprintf( stderr, "eldim : %i\n", element_dimensionality );
  fprintf( stderr, "dim   : %i\n", dimensionality );


  npy_intp dims[4];
  dims[0] = rt_dims[0];
  dims[1] = rt_dims[1];
  dims[2] = rt_dims[2];
  dims[3] = 0;
  dims[ dimensionality ] = element_dimensionality;
  dimensionality += (int)( element_dimensionality > 1 );
  fprintf( stderr, "dim   : %i\n", dimensionality );

  fprintf( stderr, "dimensionality %i\n" "dims           %i  %i  %i  %i\n" "array type     %i\n",
      dimensionality,
      dims[0],
      dims[1],
      dims[2],
      dims[3], 
      element_py_array_type );

  fprintf( stderr, "dimensionality %i\n", dimensionality );
  fprintf( stderr, "dims           %i  %i  %i  %i\n" ,
      dims[0],
      dims[1],
      dims[2],
      dims[3] );
  fprintf( stderr, "array type     %i should be %i \n", element_py_array_type, NPY_UINT8 );

  /*
  PyObject* array = PyArray_SimpleNew(
      dimensionality,
      dims,
      element_py_array_type
      );
  return array;
  */
  /*
  npy_intp strides[4];
  strides[0] = element_dimensionality;
  strides[1] = element_dimensionality*dims[0];
  strides[2] = 1; 
  return PyArray_NewFromDescr(
      &PyArray_Type, 
      PyArray_DescrFromType( element_py_array_type ),
      dimensionality,
      dims, 
      0, 
      data, 
      NPY_ARRAY_C_CONTIGUOUS,
      0 );
      */
  return PyArray_SimpleNewFromData(
      dimensionality,
      dims,
      element_py_array_type,
      data 
      );
}


