??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	
?	*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?	*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?	*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
?	?	*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?	*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?	*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
?	?	*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?	*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	? *
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
?	? *
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:? *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
?
(layer_metrics
	variables
)layer_regularization_losses

*layers
regularization_losses
+metrics
,non_trainable_variables
	trainable_variables
 
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
-layer_metrics
	variables
trainable_variables
.layer_regularization_losses

/layers
regularization_losses
0non_trainable_variables
1metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
2layer_metrics
	variables
trainable_variables
3layer_regularization_losses

4layers
regularization_losses
5non_trainable_variables
6metrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
7layer_metrics
	variables
trainable_variables
8layer_regularization_losses

9layers
regularization_losses
:non_trainable_variables
;metrics
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<layer_metrics
 	variables
!trainable_variables
=layer_regularization_losses

>layers
"regularization_losses
?non_trainable_variables
@metrics
 
 
 
?
Alayer_metrics
$	variables
%trainable_variables
Blayer_regularization_losses

Clayers
&regularization_losses
Dnon_trainable_variables
Emetrics
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@**
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

  ?D8? *.
f)R'
%__inference_signature_wrapper_4610420
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *)
f$R"
 __inference__traced_save_4610689
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *,
f'R%
#__inference__traced_restore_4610723??
?
~
)__inference_dense_5_layer_call_fn_4610624

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_46102412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?%
?
#__inference__traced_restore_4610723
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias%
!assignvariableop_2_dense_3_kernel#
assignvariableop_3_dense_3_bias%
!assignvariableop_4_dense_4_kernel#
assignvariableop_5_dense_4_bias%
!assignvariableop_6_dense_5_kernel#
assignvariableop_7_dense_5_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?0
?
D__inference_decoder_layer_call_and_return_conditional_losses_4610502

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	
?	*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_2/BiasAddq
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
dense_2/Tanh?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Tanh:y:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_3/BiasAddq
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
dense_3/Tanh?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Tanh:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_4/BiasAddq
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
dense_4/Tanh?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
?	? *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Tanh:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/BiasAddq
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
dense_5/Tanh^
reshape/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_5/Tanh:y:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????@@2
reshape/Reshape?
IdentityIdentityreshape/Reshape:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
 __inference__traced_save_4610689
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*b
_input_shapesQ
O: :	
?	:?	:
?	?	:?	:
?	?	:?	:
?	? :? : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
?	:!

_output_shapes	
:?	:&"
 
_output_shapes
:
?	?	:!

_output_shapes	
:?	:&"
 
_output_shapes
:
?	?	:!

_output_shapes	
:?	:&"
 
_output_shapes
:
?	? :!

_output_shapes	
:? :	

_output_shapes
: 
?
?
D__inference_decoder_layer_call_and_return_conditional_losses_4610378

inputs
dense_2_4610356
dense_2_4610358
dense_3_4610361
dense_3_4610363
dense_4_4610366
dense_4_4610368
dense_5_4610371
dense_5_4610373
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_4610356dense_2_4610358*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46101602!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4610361dense_3_4610363*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46101872!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4610366dense_4_4610368*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46102142!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4610371dense_5_4610373*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_46102412!
dense_5/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46102702
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
D__inference_decoder_layer_call_and_return_conditional_losses_4610332

inputs
dense_2_4610310
dense_2_4610312
dense_3_4610315
dense_3_4610317
dense_4_4610320
dense_4_4610322
dense_5_4610325
dense_5_4610327
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_4610310dense_2_4610312*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46101602!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4610315dense_3_4610317*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46101872!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4610320dense_4_4610322*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46102142!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4610325dense_5_4610327*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_46102412!
dense_5/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46102702
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_4610555

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
~
)__inference_dense_2_layer_call_fn_4610564

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46101602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_4610160

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
E
)__inference_reshape_layer_call_fn_4610642

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46102702
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_3_layer_call_fn_4610584

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46101872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
~
)__inference_dense_4_layer_call_fn_4610604

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46102142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?0
?
D__inference_decoder_layer_call_and_return_conditional_losses_4610461

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	
?	*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_2/BiasAddq
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
dense_2/Tanh?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Tanh:y:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_3/BiasAddq
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
dense_3/Tanh?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Tanh:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
dense_4/BiasAddq
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
dense_4/Tanh?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
?	? *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/Tanh:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/BiasAddq
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
dense_5/Tanh^
reshape/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_5/Tanh:y:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????@@2
reshape/Reshape?
IdentityIdentityreshape/Reshape:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_dense_4_layer_call_and_return_conditional_losses_4610595

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
D__inference_dense_4_layer_call_and_return_conditional_losses_4610214

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
)__inference_decoder_layer_call_fn_4610523

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@**
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46103322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?9
?
"__inference__wrapped_model_4610145
input_22
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resource2
.decoder_dense_5_matmul_readvariableop_resource3
/decoder_dense_5_biasadd_readvariableop_resource
identity??&decoder/dense_2/BiasAdd/ReadVariableOp?%decoder/dense_2/MatMul/ReadVariableOp?&decoder/dense_3/BiasAdd/ReadVariableOp?%decoder/dense_3/MatMul/ReadVariableOp?&decoder/dense_4/BiasAdd/ReadVariableOp?%decoder/dense_4/MatMul/ReadVariableOp?&decoder/dense_5/BiasAdd/ReadVariableOp?%decoder/dense_5/MatMul/ReadVariableOp?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	
?	*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulinput_2-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_2/BiasAdd?
decoder/dense_2/TanhTanh decoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_2/Tanh?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMuldecoder/dense_2/Tanh:y:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_3/BiasAdd?
decoder/dense_3/TanhTanh decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_3/Tanh?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMuldecoder/dense_3/Tanh:y:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_4/BiasAdd?
decoder/dense_4/TanhTanh decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
decoder/dense_4/Tanh?
%decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
?	? *
dtype02'
%decoder/dense_5/MatMul/ReadVariableOp?
decoder/dense_5/MatMulMatMuldecoder/dense_4/Tanh:y:0-decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_5/MatMul?
&decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02(
&decoder/dense_5/BiasAdd/ReadVariableOp?
decoder/dense_5/BiasAddBiasAdd decoder/dense_5/MatMul:product:0.decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_5/BiasAdd?
decoder/dense_5/TanhTanh decoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
decoder/dense_5/Tanhv
decoder/reshape/ShapeShapedecoder/dense_5/Tanh:y:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshapedecoder/dense_5/Tanh:y:0&decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????@@2
decoder/reshape/Reshape?
IdentityIdentity decoder/reshape/Reshape:output:0'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2P
&decoder/dense_2/BiasAdd/ReadVariableOp&decoder/dense_2/BiasAdd/ReadVariableOp2N
%decoder/dense_2/MatMul/ReadVariableOp%decoder/dense_2/MatMul/ReadVariableOp2P
&decoder/dense_3/BiasAdd/ReadVariableOp&decoder/dense_3/BiasAdd/ReadVariableOp2N
%decoder/dense_3/MatMul/ReadVariableOp%decoder/dense_3/MatMul/ReadVariableOp2P
&decoder/dense_4/BiasAdd/ReadVariableOp&decoder/dense_4/BiasAdd/ReadVariableOp2N
%decoder/dense_4/MatMul/ReadVariableOp%decoder/dense_4/MatMul/ReadVariableOp2P
&decoder/dense_5/BiasAdd/ReadVariableOp&decoder/dense_5/BiasAdd/ReadVariableOp2N
%decoder/dense_5/MatMul/ReadVariableOp%decoder/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_2
?
?
%__inference_signature_wrapper_4610420
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@**
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

  ?D8? *+
f&R$
"__inference__wrapped_model_46101452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_2
?
?
D__inference_decoder_layer_call_and_return_conditional_losses_4610279
input_2
dense_2_4610171
dense_2_4610173
dense_3_4610198
dense_3_4610200
dense_4_4610225
dense_4_4610227
dense_5_4610252
dense_5_4610254
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_4610171dense_2_4610173*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46101602!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4610198dense_3_4610200*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46101872!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4610225dense_4_4610227*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46102142!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4610252dense_5_4610254*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_46102412!
dense_5/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46102702
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_2
?
?
)__inference_decoder_layer_call_fn_4610351
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@**
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46103322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_2
?
?
)__inference_decoder_layer_call_fn_4610544

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@**
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46103782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
D__inference_decoder_layer_call_and_return_conditional_losses_4610304
input_2
dense_2_4610282
dense_2_4610284
dense_3_4610287
dense_3_4610289
dense_4_4610292
dense_4_4610294
dense_5_4610297
dense_5_4610299
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_4610282dense_2_4610284*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_46101602!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4610287dense_3_4610289*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_46101872!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_4610292dense_4_4610294*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_46102142!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_4610297dense_5_4610299*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_46102412!
dense_5/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_46102702
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_2
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_4610637

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????@@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_3_layer_call_and_return_conditional_losses_4610187

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
)__inference_decoder_layer_call_fn_4610397
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@@**
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

  ?D8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46103782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????
::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_2
?	
?
D__inference_dense_5_layer_call_and_return_conditional_losses_4610241

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_4610270

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????@@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_5_layer_call_and_return_conditional_losses_4610615

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
D__inference_dense_3_layer_call_and_return_conditional_losses_4610575

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????	2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????	2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????
?
reshape4
StatefulPartitionedCall:0?????????@@tensorflow/serving/predict:??
?.
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
F_default_save_signature
G__call__
*H&call_and_return_all_conditional_losses"?,
_tf_keras_network?+{"class_name": "Functional", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4096, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [64, 64]}}, "name": "reshape", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["reshape", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4096, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [64, 64]}}, "name": "reshape", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["reshape", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
I__call__
*J&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1200]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1200]}}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
O__call__
*P&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4096, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1200]}}
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [64, 64]}}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?
(layer_metrics
	variables
)layer_regularization_losses

*layers
regularization_losses
+metrics
,non_trainable_variables
	trainable_variables
G__call__
F_default_save_signature
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
,
Sserving_default"
signature_map
!:	
?	2dense_2/kernel
:?	2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-layer_metrics
	variables
trainable_variables
.layer_regularization_losses

/layers
regularization_losses
0non_trainable_variables
1metrics
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
": 
?	?	2dense_3/kernel
:?	2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2layer_metrics
	variables
trainable_variables
3layer_regularization_losses

4layers
regularization_losses
5non_trainable_variables
6metrics
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
": 
?	?	2dense_4/kernel
:?	2dense_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7layer_metrics
	variables
trainable_variables
8layer_regularization_losses

9layers
regularization_losses
:non_trainable_variables
;metrics
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
": 
?	? 2dense_5/kernel
:? 2dense_5/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<layer_metrics
 	variables
!trainable_variables
=layer_regularization_losses

>layers
"regularization_losses
?non_trainable_variables
@metrics
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Alayer_metrics
$	variables
%trainable_variables
Blayer_regularization_losses

Clayers
&regularization_losses
Dnon_trainable_variables
Emetrics
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
"__inference__wrapped_model_4610145?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_2?????????

?2?
)__inference_decoder_layer_call_fn_4610397
)__inference_decoder_layer_call_fn_4610523
)__inference_decoder_layer_call_fn_4610544
)__inference_decoder_layer_call_fn_4610351?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_decoder_layer_call_and_return_conditional_losses_4610279
D__inference_decoder_layer_call_and_return_conditional_losses_4610461
D__inference_decoder_layer_call_and_return_conditional_losses_4610502
D__inference_decoder_layer_call_and_return_conditional_losses_4610304?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_2_layer_call_fn_4610564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_2_layer_call_and_return_conditional_losses_4610555?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_3_layer_call_fn_4610584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_3_layer_call_and_return_conditional_losses_4610575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_4_layer_call_fn_4610604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_4_layer_call_and_return_conditional_losses_4610595?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_5_layer_call_fn_4610624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_5_layer_call_and_return_conditional_losses_4610615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_layer_call_fn_4610642?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_layer_call_and_return_conditional_losses_4610637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_4610420input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_4610145s0?-
&?#
!?
input_2?????????

? "5?2
0
reshape%?"
reshape?????????@@?
D__inference_decoder_layer_call_and_return_conditional_losses_4610279o8?5
.?+
!?
input_2?????????

p

 
? ")?&
?
0?????????@@
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_4610304o8?5
.?+
!?
input_2?????????

p 

 
? ")?&
?
0?????????@@
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_4610461n7?4
-?*
 ?
inputs?????????

p

 
? ")?&
?
0?????????@@
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_4610502n7?4
-?*
 ?
inputs?????????

p 

 
? ")?&
?
0?????????@@
? ?
)__inference_decoder_layer_call_fn_4610351b8?5
.?+
!?
input_2?????????

p

 
? "??????????@@?
)__inference_decoder_layer_call_fn_4610397b8?5
.?+
!?
input_2?????????

p 

 
? "??????????@@?
)__inference_decoder_layer_call_fn_4610523a7?4
-?*
 ?
inputs?????????

p

 
? "??????????@@?
)__inference_decoder_layer_call_fn_4610544a7?4
-?*
 ?
inputs?????????

p 

 
? "??????????@@?
D__inference_dense_2_layer_call_and_return_conditional_losses_4610555]/?,
%?"
 ?
inputs?????????

? "&?#
?
0??????????	
? }
)__inference_dense_2_layer_call_fn_4610564P/?,
%?"
 ?
inputs?????????

? "???????????	?
D__inference_dense_3_layer_call_and_return_conditional_losses_4610575^0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????	
? ~
)__inference_dense_3_layer_call_fn_4610584Q0?-
&?#
!?
inputs??????????	
? "???????????	?
D__inference_dense_4_layer_call_and_return_conditional_losses_4610595^0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????	
? ~
)__inference_dense_4_layer_call_fn_4610604Q0?-
&?#
!?
inputs??????????	
? "???????????	?
D__inference_dense_5_layer_call_and_return_conditional_losses_4610615^0?-
&?#
!?
inputs??????????	
? "&?#
?
0?????????? 
? ~
)__inference_dense_5_layer_call_fn_4610624Q0?-
&?#
!?
inputs??????????	
? "??????????? ?
D__inference_reshape_layer_call_and_return_conditional_losses_4610637]0?-
&?#
!?
inputs?????????? 
? ")?&
?
0?????????@@
? }
)__inference_reshape_layer_call_fn_4610642P0?-
&?#
!?
inputs?????????? 
? "??????????@@?
%__inference_signature_wrapper_4610420~;?8
? 
1?.
,
input_2!?
input_2?????????
"5?2
0
reshape%?"
reshape?????????@@