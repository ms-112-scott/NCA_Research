�	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Cos
x"T
y"T"
Ttype:

2
�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
,
Sin
x"T
y"T"
Ttype:

2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��
�
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/v
{
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/conv2d_63/kernel/v
�
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*'
_output_shapes
:�*
dtype0
�
Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_62/bias/v
|
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0�*(
shared_nameAdam/conv2d_62/kernel/v
�
+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*'
_output_shapes
:0�*
dtype0
�
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/m
{
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/conv2d_63/kernel/m
�
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*'
_output_shapes
:�*
dtype0
�
Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_62/bias/m
|
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0�*(
shared_nameAdam/conv2d_62/kernel/m
�
+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*'
_output_shapes
:0�*
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
t
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_63/bias
m
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes
:*
dtype0
�
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameconv2d_63/kernel
~
$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*'
_output_shapes
:�*
dtype0
u
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_62/bias
n
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes	
:�*
dtype0
�
conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0�*!
shared_nameconv2d_62/kernel
~
$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*'
_output_shapes
:0�*
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_392243

NoOpNoOp
�$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�$
value�$B�$ B�$
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dmodel
		optimizer

call
forward_pass
perceive

signatures*
 
0
1
2
3*
 
0
1
2
3*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 
�
layer_with_weights-0
layer-0
 layer_with_weights-1
 layer-1
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
y
'iter

(beta_1

)beta_2
	*decaym^m_m`mavbvcvdve*
)
+trace_0
,trace_1
-trace_2* 
)
.trace_0
/trace_1
0trace_2* 
)
1trace_0
2trace_1
3trace_2* 

4serving_default* 
PJ
VARIABLE_VALUEconv2d_62/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_62/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_63/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_63/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias
 ;_jit_compiled_convolution_op*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias
 B_jit_compiled_convolution_op*
 
0
1
2
3*
 
0
1
2
3*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Utrace_0* 

Vtrace_0* 
* 

0
1*

0
1*
* 
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
sm
VARIABLE_VALUEAdam/conv2d_62/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_62/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_63/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_63/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_62/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_62/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_63/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_63/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_392465
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/conv2d_62/kernel/mAdam/conv2d_62/bias/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/mAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_392523��
�
�
__inference_call_328572
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
__inference_call_328559
x
n_times(
while_input_5:0�
while_input_6:	�(
while_input_7:�
while_input_8:
identity��whileK
	Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : P
MaximumMaximumn_timesMaximum/y:output:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :l
rangeRangerange/start:output:0Maximum:z:0range/delta:output:0*#
_output_shapes
:���������N
subSubMaximum:z:0range/start:output:0*
T0*
_output_shapes
: T
floordivFloorDivsub:z:0range/delta:output:0*
T0*
_output_shapes
: O
modFloorModsub:z:0range/delta:output:0*
T0*
_output_shapes
: L

zeros_likeConst*
_output_shapes
: *
dtype0*
value	B : S
NotEqualNotEqualmod:z:0zeros_like:output:0*
T0*
_output_shapes
: J
CastCastNotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: E
addAddV2floordiv:z:0Cast:y:0*
T0*
_output_shapes
: N
zeros_like_1Const*
_output_shapes
: *
dtype0*
value	B : U
	Maximum_1Maximumadd:z:0zeros_like_1:output:0*
T0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0range/start:output:0xMaximum:z:0while_input_5while_input_6while_input_7while_input_8range/delta:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*8
_output_shapes&
$: : : :HH: : : : : : *&
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_328427*
condR
while_cond_328426*7
output_shapes&
$: : : :HH: : : : : : \
IdentityIdentitywhile:output:3^NoOp*
T0*&
_output_shapes
:HHN
NoOpNoOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:HH: : : : : 2
whilewhile:I E
&
_output_shapes
:HH

_user_specified_namex:?;

_output_shapes
: 
!
_user_specified_name	n_times
�5
2
__inference_perceive_328093
x
identityJ
Cos/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
CosCosCos/x:output:0*
T0*
_output_shapes
: J
Sin/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
SinSinSin/x:output:0*
T0*
_output_shapes
: z
mul/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_1MulSin:y:0mul_1/y:output:0*
T0*
_output_shapes

:G
subSubmul:z:0	mul_1:z:0*
T0*
_output_shapes

:|
mul_2/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_3MulCos:y:0mul_3/y:output:0*
T0*
_output_shapes

:K
addAddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:z
stackConst*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis���������l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_sliceStridedSlicestack_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: e
Repeat/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0**
_output_shapes
:Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/4Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0**
_output_shapes
:d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:z
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*&
_output_shapes
:h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
b
IdentityIdentitydepthwise:output:0*
T0*/
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392174
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�

�
E__inference_conv2d_63_layer_call_and_return_conditional_losses_391999

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_ca_model_32_layer_call_fn_392269
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392174w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
$__inference_signature_wrapper_392243
input_1"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_391965w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
__inference_forward_pass_328610
xQ
6sequential_31_conv2d_62_conv2d_readvariableop_resource:0�F
7sequential_31_conv2d_62_biasadd_readvariableop_resource:	�Q
6sequential_31_conv2d_63_conv2d_readvariableop_resource:�E
7sequential_31_conv2d_63_biasadd_readvariableop_resource:
identity��.sequential_31/conv2d_62/BiasAdd/ReadVariableOp�-sequential_31/conv2d_62/Conv2D/ReadVariableOp�.sequential_31/conv2d_63/BiasAdd/ReadVariableOp�-sequential_31/conv2d_63/Conv2D/ReadVariableOp�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:HH0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_perceive_328504�
-sequential_31/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
sequential_31/conv2d_62/Conv2DConv2DPartitionedCall:output:05sequential_31/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:HH�*
paddingVALID*
strides
�
.sequential_31/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_31/conv2d_62/BiasAddBiasAdd'sequential_31/conv2d_62/Conv2D:output:06sequential_31/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:HH��
sequential_31/conv2d_62/ReluRelu(sequential_31/conv2d_62/BiasAdd:output:0*
T0*'
_output_shapes
:HH��
-sequential_31/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
sequential_31/conv2d_63/Conv2DConv2D*sequential_31/conv2d_62/Relu:activations:05sequential_31/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:HH*
paddingVALID*
strides
�
.sequential_31/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_31/conv2d_63/BiasAddBiasAdd'sequential_31/conv2d_63/Conv2D:output:06sequential_31/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:HHj
addAddV2x(sequential_31/conv2d_63/BiasAdd:output:0*
T0*&
_output_shapes
:HHU
IdentityIdentityadd:z:0^NoOp*
T0*&
_output_shapes
:HH�
NoOpNoOp/^sequential_31/conv2d_62/BiasAdd/ReadVariableOp.^sequential_31/conv2d_62/Conv2D/ReadVariableOp/^sequential_31/conv2d_63/BiasAdd/ReadVariableOp.^sequential_31/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:HH: : : : 2`
.sequential_31/conv2d_62/BiasAdd/ReadVariableOp.sequential_31/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_62/Conv2D/ReadVariableOp-sequential_31/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_31/conv2d_63/BiasAdd/ReadVariableOp.sequential_31/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_63/Conv2D/ReadVariableOp-sequential_31/conv2d_63/Conv2D/ReadVariableOp:I E
&
_output_shapes
:HH

_user_specified_namex
�
�
,__inference_ca_model_32_layer_call_fn_392198
input_1"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392174w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
while_cond_328426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_less_maximum4
0while_while_cond_328426___redundant_placeholder04
0while_while_cond_328426___redundant_placeholder14
0while_while_cond_328426___redundant_placeholder24
0while_while_cond_328426___redundant_placeholder34
0while_while_cond_328426___redundant_placeholder4
while_identity
Z

while/LessLesswhile_placeholderwhile_less_maximum*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : :HH: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:HH:

_output_shapes
: :	

_output_shapes
:
�
�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392118
conv2d_62_input+
conv2d_62_392107:0�
conv2d_62_392109:	�+
conv2d_63_392112:�
conv2d_63_392114:
identity��!conv2d_62/StatefulPartitionedCall�!conv2d_63/StatefulPartitionedCall�
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputconv2d_62_392107conv2d_62_392109*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_62_layer_call_and_return_conditional_losses_391983�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_392112conv2d_63_392114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_63_layer_call_and_return_conditional_losses_391999�
IdentityIdentity*conv2d_63/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:` \
/
_output_shapes
:���������0
)
_user_specified_nameconv2d_62_input
�5
2
__inference_perceive_328747
x
identityJ
Cos/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
CosCosCos/x:output:0*
T0*
_output_shapes
: J
Sin/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
SinSinSin/x:output:0*
T0*
_output_shapes
: z
mul/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_1MulSin:y:0mul_1/y:output:0*
T0*
_output_shapes

:G
subSubmul:z:0	mul_1:z:0*
T0*
_output_shapes

:|
mul_2/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_3MulCos:y:0mul_3/y:output:0*
T0*
_output_shapes

:K
addAddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:z
stackConst*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis���������l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_sliceStridedSlicestack_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: e
Repeat/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0**
_output_shapes
:Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/4Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0**
_output_shapes
:d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:z
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*&
_output_shapes
:h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*&
_output_shapes
:HH0*
paddingSAME*
strides
Y
IdentityIdentitydepthwise:output:0*
T0*&
_output_shapes
:HH0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:HH:I E
&
_output_shapes
:HH

_user_specified_namex
�
�
__inference_forward_pass_328110
xQ
6sequential_31_conv2d_62_conv2d_readvariableop_resource:0�F
7sequential_31_conv2d_62_biasadd_readvariableop_resource:	�Q
6sequential_31_conv2d_63_conv2d_readvariableop_resource:�E
7sequential_31_conv2d_63_biasadd_readvariableop_resource:
identity��.sequential_31/conv2d_62/BiasAdd/ReadVariableOp�-sequential_31/conv2d_62/Conv2D/ReadVariableOp�.sequential_31/conv2d_63/BiasAdd/ReadVariableOp�-sequential_31/conv2d_63/Conv2D/ReadVariableOp�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_perceive_328093�
-sequential_31/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
sequential_31/conv2d_62/Conv2DConv2DPartitionedCall:output:05sequential_31/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
.sequential_31/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_31/conv2d_62/BiasAddBiasAdd'sequential_31/conv2d_62/Conv2D:output:06sequential_31/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
sequential_31/conv2d_62/ReluRelu(sequential_31/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
-sequential_31/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
sequential_31/conv2d_63/Conv2DConv2D*sequential_31/conv2d_62/Relu:activations:05sequential_31/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
.sequential_31/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_31/conv2d_63/BiasAddBiasAdd'sequential_31/conv2d_63/Conv2D:output:06sequential_31/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������s
addAddV2x(sequential_31/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:���������^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp/^sequential_31/conv2d_62/BiasAdd/ReadVariableOp.^sequential_31/conv2d_62/Conv2D/ReadVariableOp/^sequential_31/conv2d_63/BiasAdd/ReadVariableOp.^sequential_31/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2`
.sequential_31/conv2d_62/BiasAdd/ReadVariableOp.sequential_31/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_62/Conv2D/ReadVariableOp-sequential_31/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_31/conv2d_63/BiasAdd/ReadVariableOp.sequential_31/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_63/Conv2D/ReadVariableOp-sequential_31/conv2d_63/Conv2D/ReadVariableOp:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
__inference_forward_pass_328521
xQ
6sequential_31_conv2d_62_conv2d_readvariableop_resource:0�F
7sequential_31_conv2d_62_biasadd_readvariableop_resource:	�Q
6sequential_31_conv2d_63_conv2d_readvariableop_resource:�E
7sequential_31_conv2d_63_biasadd_readvariableop_resource:
identity��.sequential_31/conv2d_62/BiasAdd/ReadVariableOp�-sequential_31/conv2d_62/Conv2D/ReadVariableOp�.sequential_31/conv2d_63/BiasAdd/ReadVariableOp�-sequential_31/conv2d_63/Conv2D/ReadVariableOp�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:HH0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_perceive_328504�
-sequential_31/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
sequential_31/conv2d_62/Conv2DConv2DPartitionedCall:output:05sequential_31/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:HH�*
paddingVALID*
strides
�
.sequential_31/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_31/conv2d_62/BiasAddBiasAdd'sequential_31/conv2d_62/Conv2D:output:06sequential_31/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:HH��
sequential_31/conv2d_62/ReluRelu(sequential_31/conv2d_62/BiasAdd:output:0*
T0*'
_output_shapes
:HH��
-sequential_31/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
sequential_31/conv2d_63/Conv2DConv2D*sequential_31/conv2d_62/Relu:activations:05sequential_31/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:HH*
paddingVALID*
strides
�
.sequential_31/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_31/conv2d_63/BiasAddBiasAdd'sequential_31/conv2d_63/Conv2D:output:06sequential_31/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:HHj
addAddV2x(sequential_31/conv2d_63/BiasAdd:output:0*
T0*&
_output_shapes
:HHU
IdentityIdentityadd:z:0^NoOp*
T0*&
_output_shapes
:HH�
NoOpNoOp/^sequential_31/conv2d_62/BiasAdd/ReadVariableOp.^sequential_31/conv2d_62/Conv2D/ReadVariableOp/^sequential_31/conv2d_63/BiasAdd/ReadVariableOp.^sequential_31/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:HH: : : : 2`
.sequential_31/conv2d_62/BiasAdd/ReadVariableOp.sequential_31/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_62/Conv2D/ReadVariableOp-sequential_31/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_31/conv2d_63/BiasAdd/ReadVariableOp.sequential_31/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_63/Conv2D/ReadVariableOp-sequential_31/conv2d_63/Conv2D/ReadVariableOp:I E
&
_output_shapes
:HH

_user_specified_namex
�
�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392295
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�B
�

"__inference__traced_restore_392523
file_prefix<
!assignvariableop_conv2d_62_kernel:0�0
!assignvariableop_1_conv2d_62_bias:	�>
#assignvariableop_2_conv2d_63_kernel:�/
!assignvariableop_3_conv2d_63_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: E
*assignvariableop_8_adam_conv2d_62_kernel_m:0�7
(assignvariableop_9_adam_conv2d_62_bias_m:	�F
+assignvariableop_10_adam_conv2d_63_kernel_m:�7
)assignvariableop_11_adam_conv2d_63_bias_m:F
+assignvariableop_12_adam_conv2d_62_kernel_v:0�8
)assignvariableop_13_adam_conv2d_62_bias_v:	�F
+assignvariableop_14_adam_conv2d_63_kernel_v:�7
)assignvariableop_15_adam_conv2d_63_bias_v:
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_62_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_62_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_63_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_63_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_adam_conv2d_62_kernel_mIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_conv2d_62_bias_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_adam_conv2d_63_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_conv2d_63_bias_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_adam_conv2d_62_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_conv2d_62_bias_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_conv2d_63_kernel_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_conv2d_63_bias_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392355

inputsC
(conv2d_62_conv2d_readvariableop_resource:0�8
)conv2d_62_biasadd_readvariableop_resource:	�C
(conv2d_63_conv2d_readvariableop_resource:�7
)conv2d_63_biasadd_readvariableop_resource:
identity�� conv2d_62/BiasAdd/ReadVariableOp�conv2d_62/Conv2D/ReadVariableOp� conv2d_63/BiasAdd/ReadVariableOp�conv2d_63/Conv2D/ReadVariableOp�
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
conv2d_62/Conv2DConv2Dinputs'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������q
IdentityIdentityconv2d_63/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
.__inference_sequential_31_layer_call_fn_392321

inputs"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_31_layer_call_and_return_conditional_losses_392066w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�5
2
__inference_perceive_328504
x
identityJ
Cos/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
CosCosCos/x:output:0*
T0*
_output_shapes
: J
Sin/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
SinSinSin/x:output:0*
T0*
_output_shapes
: z
mul/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_1MulSin:y:0mul_1/y:output:0*
T0*
_output_shapes

:G
subSubmul:z:0	mul_1:z:0*
T0*
_output_shapes

:|
mul_2/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_3MulCos:y:0mul_3/y:output:0*
T0*
_output_shapes

:K
addAddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:z
stackConst*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis���������l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_sliceStridedSlicestack_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: e
Repeat/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0**
_output_shapes
:Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/4Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0**
_output_shapes
:d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:z
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*&
_output_shapes
:h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*&
_output_shapes
:HH0*
paddingSAME*
strides
Y
IdentityIdentitydepthwise:output:0*
T0*&
_output_shapes
:HH0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:HH:I E
&
_output_shapes
:HH

_user_specified_namex
�
�
,__inference_ca_model_32_layer_call_fn_392146
input_1"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392135w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
!__inference__wrapped_model_391965
input_1-
ca_model_32_391955:0�!
ca_model_32_391957:	�-
ca_model_32_391959:� 
ca_model_32_391961:
identity��#ca_model_32/StatefulPartitionedCall�
#ca_model_32/StatefulPartitionedCallStatefulPartitionedCallinput_1ca_model_32_391955ca_model_32_391957ca_model_32_391959ca_model_32_391961*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� * 
fR
__inference_call_328121�
IdentityIdentity,ca_model_32/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������l
NoOpNoOp$^ca_model_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2J
#ca_model_32/StatefulPartitionedCall#ca_model_32/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
E__inference_conv2d_63_layer_call_and_return_conditional_losses_392394

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_call_328121
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�*
�
__inference__traced_save_392465
file_prefix/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :0�:�:�:: : : : :0�:�:�::0�:�:�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:0�:!

_output_shapes	
:�:-)
'
_output_shapes
:�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-	)
'
_output_shapes
:0�:!


_output_shapes	
:�:-)
'
_output_shapes
:�: 

_output_shapes
::-)
'
_output_shapes
:0�:!

_output_shapes	
:�:-)
'
_output_shapes
:�: 

_output_shapes
::

_output_shapes
: 
�
�
.__inference_sequential_31_layer_call_fn_392017
conv2d_62_input"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_31_layer_call_and_return_conditional_losses_392006w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������0
)
_user_specified_nameconv2d_62_input
�
�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392338

inputsC
(conv2d_62_conv2d_readvariableop_resource:0�8
)conv2d_62_biasadd_readvariableop_resource:	�C
(conv2d_63_conv2d_readvariableop_resource:�7
)conv2d_63_biasadd_readvariableop_resource:
identity�� conv2d_62/BiasAdd/ReadVariableOp�conv2d_62/Conv2D/ReadVariableOp� conv2d_63/BiasAdd/ReadVariableOp�conv2d_63/Conv2D/ReadVariableOp�
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
conv2d_62/Conv2DConv2Dinputs'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������q
IdentityIdentityconv2d_63/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392006

inputs+
conv2d_62_391984:0�
conv2d_62_391986:	�+
conv2d_63_392000:�
conv2d_63_392002:
identity��!conv2d_62/StatefulPartitionedCall�!conv2d_63/StatefulPartitionedCall�
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_62_391984conv2d_62_391986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_62_layer_call_and_return_conditional_losses_391983�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_392000conv2d_63_392002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_63_layer_call_and_return_conditional_losses_391999�
IdentityIdentity*conv2d_63/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392224
input_1"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
__inference_forward_pass_328629
xQ
6sequential_31_conv2d_62_conv2d_readvariableop_resource:0�F
7sequential_31_conv2d_62_biasadd_readvariableop_resource:	�Q
6sequential_31_conv2d_63_conv2d_readvariableop_resource:�E
7sequential_31_conv2d_63_biasadd_readvariableop_resource:
identity��.sequential_31/conv2d_62/BiasAdd/ReadVariableOp�-sequential_31/conv2d_62/Conv2D/ReadVariableOp�.sequential_31/conv2d_63/BiasAdd/ReadVariableOp�-sequential_31/conv2d_63/Conv2D/ReadVariableOp�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_perceive_328093�
-sequential_31/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
sequential_31/conv2d_62/Conv2DConv2DPartitionedCall:output:05sequential_31/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
.sequential_31/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_31/conv2d_62/BiasAddBiasAdd'sequential_31/conv2d_62/Conv2D:output:06sequential_31/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
sequential_31/conv2d_62/ReluRelu(sequential_31/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
-sequential_31/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
sequential_31/conv2d_63/Conv2DConv2D*sequential_31/conv2d_62/Relu:activations:05sequential_31/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
.sequential_31/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_31/conv2d_63/BiasAddBiasAdd'sequential_31/conv2d_63/Conv2D:output:06sequential_31/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������s
addAddV2x(sequential_31/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:���������^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp/^sequential_31/conv2d_62/BiasAdd/ReadVariableOp.^sequential_31/conv2d_62/Conv2D/ReadVariableOp/^sequential_31/conv2d_63/BiasAdd/ReadVariableOp.^sequential_31/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 2`
.sequential_31/conv2d_62/BiasAdd/ReadVariableOp.sequential_31/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_62/Conv2D/ReadVariableOp-sequential_31/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_31/conv2d_63/BiasAdd/ReadVariableOp.sequential_31/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_63/Conv2D/ReadVariableOp-sequential_31/conv2d_63/Conv2D/ReadVariableOp:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392066

inputs+
conv2d_62_392055:0�
conv2d_62_392057:	�+
conv2d_63_392060:�
conv2d_63_392062:
identity��!conv2d_62/StatefulPartitionedCall�!conv2d_63/StatefulPartitionedCall�
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_62_392055conv2d_62_392057*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_62_layer_call_and_return_conditional_losses_391983�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_392060conv2d_63_392062*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_63_layer_call_and_return_conditional_losses_391999�
IdentityIdentity*conv2d_63/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�5
2
__inference_perceive_328688
x
identityJ
Cos/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
CosCosCos/x:output:0*
T0*
_output_shapes
: J
Sin/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
SinSinSin/x:output:0*
T0*
_output_shapes
: z
mul/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_1MulSin:y:0mul_1/y:output:0*
T0*
_output_shapes

:G
subSubmul:z:0	mul_1:z:0*
T0*
_output_shapes

:|
mul_2/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_3MulCos:y:0mul_3/y:output:0*
T0*
_output_shapes

:K
addAddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:z
stackConst*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis���������l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_sliceStridedSlicestack_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: e
Repeat/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0**
_output_shapes
:Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/4Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0**
_output_shapes
:d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:z
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*&
_output_shapes
:h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
Y
IdentityIdentitydepthwise:output:0*
T0*&
_output_shapes
:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::I E
&
_output_shapes
:

_user_specified_namex
�
�
*__inference_conv2d_63_layer_call_fn_392384

inputs"
unknown:�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_63_layer_call_and_return_conditional_losses_391999w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�5
2
__inference_perceive_328806
x
identityJ
Cos/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
CosCosCos/x:output:0*
T0*
_output_shapes
: J
Sin/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ;
SinSinSin/x:output:0*
T0*
_output_shapes
: z
mul/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_1MulSin:y:0mul_1/y:output:0*
T0*
_output_shapes

:G
subSubmul:z:0	mul_1:z:0*
T0*
_output_shapes

:|
mul_2/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �       >  ��      �>   �       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   �  ��   �               >  �>   >P
mul_3MulCos:y:0mul_3/y:output:0*
T0*
_output_shapes

:K
addAddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:z
stackConst*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  �?                �
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis���������l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_sliceStridedSlicestack_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: e
Repeat/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0**
_output_shapes
:Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/4Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0**
_output_shapes
:d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:z
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*&
_output_shapes
:h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
b
IdentityIdentitydepthwise:output:0*
T0*/
_output_shapes
:���������0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
.__inference_sequential_31_layer_call_fn_392308

inputs"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_31_layer_call_and_return_conditional_losses_392006w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392282
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
__inference_forward_pass_328591
xQ
6sequential_31_conv2d_62_conv2d_readvariableop_resource:0�F
7sequential_31_conv2d_62_biasadd_readvariableop_resource:	�Q
6sequential_31_conv2d_63_conv2d_readvariableop_resource:�E
7sequential_31_conv2d_63_biasadd_readvariableop_resource:
identity��.sequential_31/conv2d_62/BiasAdd/ReadVariableOp�-sequential_31/conv2d_62/Conv2D/ReadVariableOp�.sequential_31/conv2d_63/BiasAdd/ReadVariableOp�-sequential_31/conv2d_63/Conv2D/ReadVariableOp�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_perceive_328093�
-sequential_31/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_62_conv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
sequential_31/conv2d_62/Conv2DConv2DPartitionedCall:output:05sequential_31/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
.sequential_31/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_31/conv2d_62/BiasAddBiasAdd'sequential_31/conv2d_62/Conv2D:output:06sequential_31/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��
sequential_31/conv2d_62/ReluRelu(sequential_31/conv2d_62/BiasAdd:output:0*
T0*'
_output_shapes
:��
-sequential_31/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_63_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
sequential_31/conv2d_63/Conv2DConv2D*sequential_31/conv2d_62/Relu:activations:05sequential_31/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
.sequential_31/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_31/conv2d_63/BiasAddBiasAdd'sequential_31/conv2d_63/Conv2D:output:06sequential_31/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:j
addAddV2x(sequential_31/conv2d_63/BiasAdd:output:0*
T0*&
_output_shapes
:U
IdentityIdentityadd:z:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp/^sequential_31/conv2d_62/BiasAdd/ReadVariableOp.^sequential_31/conv2d_62/Conv2D/ReadVariableOp/^sequential_31/conv2d_63/BiasAdd/ReadVariableOp.^sequential_31/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : 2`
.sequential_31/conv2d_62/BiasAdd/ReadVariableOp.sequential_31/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_62/Conv2D/ReadVariableOp-sequential_31/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_31/conv2d_63/BiasAdd/ReadVariableOp.sequential_31/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_63/Conv2D/ReadVariableOp-sequential_31/conv2d_63/Conv2D/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
�
�
E__inference_conv2d_62_layer_call_and_return_conditional_losses_391983

inputs9
conv2d_readvariableop_resource:0�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392211
input_1"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392135
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
while_body_328427
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_maximum_0)
while_328522_0:0�
while_328524_0:	�)
while_328526_0:�
while_328528_0:
while_add_range_delta_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_maximum'
while_328522:0�
while_328524:	�'
while_328526:�
while_328528:
while_add_range_delta��while/StatefulPartitionedCall�
while/StatefulPartitionedCallStatefulPartitionedCallwhile_placeholder_1while_328522_0while_328524_0while_328526_0while_328528_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:HH*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328521_
	while/addAddV2while_placeholderwhile_add_range_delta_0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity&while/StatefulPartitionedCall:output:0^while/NoOp*
T0*&
_output_shapes
:HHl

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_328522while_328522_0"
while_328524while_328524_0"
while_328526while_328526_0"
while_328528while_328528_0"0
while_add_range_deltawhile_add_range_delta_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0" 
while_maximumwhile_maximum_0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : :HH: : : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:HH:

_output_shapes
: :	

_output_shapes
: 
�
�
,__inference_ca_model_32_layer_call_fn_392256
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392135w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
.__inference_sequential_31_layer_call_fn_392090
conv2d_62_input"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_31_layer_call_and_return_conditional_losses_392066w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������0
)
_user_specified_nameconv2d_62_input
�
�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392104
conv2d_62_input+
conv2d_62_392093:0�
conv2d_62_392095:	�+
conv2d_63_392098:�
conv2d_63_392100:
identity��!conv2d_62/StatefulPartitionedCall�!conv2d_63/StatefulPartitionedCall�
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputconv2d_62_392093conv2d_62_392095*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_62_layer_call_and_return_conditional_losses_391983�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_392098conv2d_63_392100*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_63_layer_call_and_return_conditional_losses_391999�
IdentityIdentity*conv2d_63/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������0: : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:` \
/
_output_shapes
:���������0
)
_user_specified_nameconv2d_62_input
�
�
__inference_call_328408
x"
unknown:0�
	unknown_0:	�$
	unknown_1:�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference_forward_pass_328110n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:I E
&
_output_shapes
:

_user_specified_namex
�
�
E__inference_conv2d_62_layer_call_and_return_conditional_losses_392375

inputs9
conv2d_readvariableop_resource:0�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
*__inference_conv2d_62_layer_call_fn_392364

inputs"
unknown:0�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_62_layer_call_and_return_conditional_losses_391983x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������D
output_18
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dmodel
		optimizer

call
forward_pass
perceive

signatures"
_tf_keras_model
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
,__inference_ca_model_32_layer_call_fn_392146
,__inference_ca_model_32_layer_call_fn_392256
,__inference_ca_model_32_layer_call_fn_392269
,__inference_ca_model_32_layer_call_fn_392198�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392282
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392295
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392211
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392224�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�B�
!__inference__wrapped_model_391965input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
layer_with_weights-0
layer-0
 layer_with_weights-1
 layer-1
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
'iter

(beta_1

)beta_2
	*decaym^m_m`mavbvcvdve"
	optimizer
�
+trace_0
,trace_1
-trace_22�
__inference_call_328408
__inference_call_328559
__inference_call_328572�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z+trace_0z,trace_1z-trace_2
�
.trace_0
/trace_1
0trace_22�
__inference_forward_pass_328591
__inference_forward_pass_328610
__inference_forward_pass_328629�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z.trace_0z/trace_1z0trace_2
�
1trace_0
2trace_1
3trace_22�
__inference_perceive_328688
__inference_perceive_328747
__inference_perceive_328806�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z1trace_0z2trace_1z3trace_2
,
4serving_default"
signature_map
+:)0�2conv2d_62/kernel
:�2conv2d_62/bias
+:)�2conv2d_63/kernel
:2conv2d_63/bias
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_ca_model_32_layer_call_fn_392146input_1"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
,__inference_ca_model_32_layer_call_fn_392256x"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
,__inference_ca_model_32_layer_call_fn_392269x"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
,__inference_ca_model_32_layer_call_fn_392198input_1"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392282x"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392295x"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392211input_1"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392224input_1"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias
 ;_jit_compiled_convolution_op"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias
 B_jit_compiled_convolution_op"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32�
.__inference_sequential_31_layer_call_fn_392017
.__inference_sequential_31_layer_call_fn_392308
.__inference_sequential_31_layer_call_fn_392321
.__inference_sequential_31_layer_call_fn_392090�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
�
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392338
I__inference_sequential_31_layer_call_and_return_conditional_losses_392355
I__inference_sequential_31_layer_call_and_return_conditional_losses_392104
I__inference_sequential_31_layer_call_and_return_conditional_losses_392118�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
�B�
__inference_call_328408x"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_call_328559xn_times"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_call_328572x"�
���
FullArgSpec#
args�
jself
jx
	jn_times
varargs
 
varkw
 
defaults�
`

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_forward_pass_328591x"�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_forward_pass_328610x"�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_forward_pass_328629x"�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_perceive_328688x"�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_perceive_328747x"�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_perceive_328806x"�
���
FullArgSpec!
args�
jself
jx
jangle
varargs
 
varkw
 
defaults�
	Y        

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_392243input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
Utrace_02�
*__inference_conv2d_62_layer_call_fn_392364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0
�
Vtrace_02�
E__inference_conv2d_62_layer_call_and_return_conditional_losses_392375�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
\trace_02�
*__inference_conv2d_63_layer_call_fn_392384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0
�
]trace_02�
E__inference_conv2d_63_layer_call_and_return_conditional_losses_392394�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_31_layer_call_fn_392017conv2d_62_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_31_layer_call_fn_392308inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_31_layer_call_fn_392321inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_31_layer_call_fn_392090conv2d_62_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392338inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392355inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392104conv2d_62_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392118conv2d_62_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_conv2d_62_layer_call_fn_392364inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_62_layer_call_and_return_conditional_losses_392375inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_conv2d_63_layer_call_fn_392384inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_63_layer_call_and_return_conditional_losses_392394inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0:.0�2Adam/conv2d_62/kernel/m
": �2Adam/conv2d_62/bias/m
0:.�2Adam/conv2d_63/kernel/m
!:2Adam/conv2d_63/bias/m
0:.0�2Adam/conv2d_62/kernel/v
": �2Adam/conv2d_62/bias/v
0:.�2Adam/conv2d_63/kernel/v
!:2Adam/conv2d_63/bias/v�
!__inference__wrapped_model_391965}8�5
.�+
)�&
input_1���������
� ";�8
6
output_1*�'
output_1����������
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392211�L�I
2�/
)�&
input_1���������
`
�

trainingp "-�*
#� 
0���������
� �
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392224�L�I
2�/
)�&
input_1���������
`
�

trainingp"-�*
#� 
0���������
� �
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392282}F�C
,�)
#� 
x���������
`
�

trainingp "-�*
#� 
0���������
� �
G__inference_ca_model_32_layer_call_and_return_conditional_losses_392295}F�C
,�)
#� 
x���������
`
�

trainingp"-�*
#� 
0���������
� �
,__inference_ca_model_32_layer_call_fn_392146vL�I
2�/
)�&
input_1���������
`
�

trainingp " �����������
,__inference_ca_model_32_layer_call_fn_392198vL�I
2�/
)�&
input_1���������
`
�

trainingp" �����������
,__inference_ca_model_32_layer_call_fn_392256pF�C
,�)
#� 
x���������
`
�

trainingp " �����������
,__inference_ca_model_32_layer_call_fn_392269pF�C
,�)
#� 
x���������
`
�

trainingp" ����������i
__inference_call_328408N-�*
#� 
�
x
`
� "�w
__inference_call_328559\;�8
1�.
�
xHH
�
n_times 
� "�HH{
__inference_call_328572`6�3
,�)
#� 
x���������
`
� " �����������
E__inference_conv2d_62_layer_call_and_return_conditional_losses_392375m7�4
-�*
(�%
inputs���������0
� ".�+
$�!
0����������
� �
*__inference_conv2d_62_layer_call_fn_392364`7�4
-�*
(�%
inputs���������0
� "!������������
E__inference_conv2d_63_layer_call_and_return_conditional_losses_392394m8�5
.�+
)�&
inputs����������
� "-�*
#� 
0���������
� �
*__inference_conv2d_63_layer_call_fn_392384`8�5
.�+
)�&
inputs����������
� " ����������x
__inference_forward_pass_328591U4�1
*�'
�
x
	Y        
� "�x
__inference_forward_pass_328610U4�1
*�'
�
xHH
	Y        
� "�HH�
__inference_forward_pass_328629g=�:
3�0
#� 
x���������
	Y        
� " ����������n
__inference_perceive_328688O4�1
*�'
�
x
	Y        
� "�0n
__inference_perceive_328747O4�1
*�'
�
xHH
	Y        
� "�HH0�
__inference_perceive_328806a=�:
3�0
#� 
x���������
	Y        
� " ����������0�
I__inference_sequential_31_layer_call_and_return_conditional_losses_392104H�E
>�;
1�.
conv2d_62_input���������0
p 

 
� "-�*
#� 
0���������
� �
I__inference_sequential_31_layer_call_and_return_conditional_losses_392118H�E
>�;
1�.
conv2d_62_input���������0
p

 
� "-�*
#� 
0���������
� �
I__inference_sequential_31_layer_call_and_return_conditional_losses_392338v?�<
5�2
(�%
inputs���������0
p 

 
� "-�*
#� 
0���������
� �
I__inference_sequential_31_layer_call_and_return_conditional_losses_392355v?�<
5�2
(�%
inputs���������0
p

 
� "-�*
#� 
0���������
� �
.__inference_sequential_31_layer_call_fn_392017rH�E
>�;
1�.
conv2d_62_input���������0
p 

 
� " �����������
.__inference_sequential_31_layer_call_fn_392090rH�E
>�;
1�.
conv2d_62_input���������0
p

 
� " �����������
.__inference_sequential_31_layer_call_fn_392308i?�<
5�2
(�%
inputs���������0
p 

 
� " �����������
.__inference_sequential_31_layer_call_fn_392321i?�<
5�2
(�%
inputs���������0
p

 
� " �����������
$__inference_signature_wrapper_392243�C�@
� 
9�6
4
input_1)�&
input_1���������";�8
6
output_1*�'
output_1���������