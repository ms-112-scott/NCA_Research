ґЛ	
–£
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ы
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
Л
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
≥
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8≠ƒ
В
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_17/bias/v
{
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes
:*
dtype0
У
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/conv2d_17/kernel/v
М
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*'
_output_shapes
:А*
dtype0
Г
Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_16/bias/v
|
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
_output_shapes	
:А*
dtype0
У
Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0А*(
shared_nameAdam/conv2d_16/kernel/v
М
+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*'
_output_shapes
:0А*
dtype0
В
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_17/bias/m
{
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes
:*
dtype0
У
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/conv2d_17/kernel/m
М
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*'
_output_shapes
:А*
dtype0
Г
Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_16/bias/m
|
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
_output_shapes	
:А*
dtype0
У
Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0А*(
shared_nameAdam/conv2d_16/kernel/m
М
+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*'
_output_shapes
:0А*
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
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0
Е
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:А*
dtype0
u
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_16/bias
n
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes	
:А*
dtype0
Е
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0А*!
shared_nameconv2d_16/kernel
~
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*'
_output_shapes
:0А*
dtype0
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
Й
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_146709

NoOpNoOp
ю$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*є$
valueѓ$Bђ$ B•$
В
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
∞
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
ё
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
VARIABLE_VALUEconv2d_16/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_16/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_17/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_17/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
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
»
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias
 ;_jit_compiled_convolution_op*
»
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
У
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
У
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
У
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
VARIABLE_VALUEAdam/conv2d_16/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_16/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_17/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_17/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_16/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_16/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_17/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_17/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOpConst*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_146931
ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/mAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/v*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_146989оо
т
†
*__inference_conv2d_17_layer_call_fn_146850

inputs"
unknown:А
	unknown_0:
identityИҐStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146465w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Г5
2
__inference_perceive_113686
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
value4B2"$   Њ       >  АЊ      А>   Њ       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$   Њ       >  АЊ      А>   Њ       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$                  А?                Е
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  А?                З
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis€€€€€€€€€l
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
valueB"            В
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
value	B :М
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
value	B :м
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:Д
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
valueB:т
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
valueB:ь
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
valueB:ш
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
value	B : ≈
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
valueB"      Щ
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
≠

€
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146465

inputs9
conv2d_readvariableop_resource:А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
х
 
H__inference_sequential_8_layer_call_and_return_conditional_losses_146532

inputs+
conv2d_16_146521:0А
conv2d_16_146523:	А+
conv2d_17_146526:А
conv2d_17_146528:
identityИҐ!conv2d_16/StatefulPartitionedCallҐ!conv2d_17/StatefulPartitionedCallА
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_146521conv2d_16_146523*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146449£
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_146526conv2d_17_146528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146465Б
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€О
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
І5
2
__inference_perceive_113032
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
value4B2"$   Њ       >  АЊ      А>   Њ       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$   Њ       >  АЊ      А>   Њ       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$                  А?                Е
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  А?                З
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis€€€€€€€€€l
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
valueB"            В
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
value	B :М
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
value	B :м
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:Д
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
valueB:т
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
valueB:ь
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
valueB:ш
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
value	B : ≈
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
valueB"      Ґ
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€0*
paddingSAME*
strides
b
IdentityIdentitydepthwise:output:0*
T0*/
_output_shapes
:€€€€€€€€€0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Ю
ь
while_body_113366
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_maximum_0)
while_113461_0:0А
while_113463_0:	А)
while_113465_0:А
while_113467_0:
while_add_range_delta_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_maximum'
while_113461:0А
while_113463:	А'
while_113465:А
while_113467:
while_add_range_deltaИҐwhile/StatefulPartitionedCallщ
while/StatefulPartitionedCallStatefulPartitionedCallwhile_placeholder_1while_113461_0while_113463_0while_113465_0while_113467_0*
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
GPU2*0J 8В *(
f#R!
__inference_forward_pass_113460_
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
: В
while/Identity_3Identity&while/StatefulPartitionedCall:output:0^while/NoOp*
T0*&
_output_shapes
:HHl

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_113461while_113461_0"
while_113463while_113463_0"
while_113465while_113465_0"
while_113467while_113467_0"0
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
Е
о
H__inference_sequential_8_layer_call_and_return_conditional_losses_146804

inputsC
(conv2d_16_conv2d_readvariableop_resource:0А8
)conv2d_16_biasadd_readvariableop_resource:	АC
(conv2d_17_conv2d_readvariableop_resource:А7
)conv2d_17_biasadd_readvariableop_resource:
identityИҐ conv2d_16/BiasAdd/ReadVariableOpҐconv2d_16/Conv2D/ReadVariableOpҐ conv2d_17/BiasAdd/ReadVariableOpҐconv2d_17/Conv2D/ReadVariableOpС
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0ѓ
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
З
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аm
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€АС
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ƒ
conv2d_17/Conv2DConv2Dconv2d_16/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
Ж
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ы
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€q
IdentityIdentityconv2d_17/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€–
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
µ
ч
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146761
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
«
э
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146677
input_1"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
п
м
-__inference_sequential_8_layer_call_fn_146483
conv2d_16_input"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146472w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€0
)
_user_specified_nameconv2d_16_input
Ѕ
№
+__inference_ca_model_8_layer_call_fn_146735
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146640w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
µ
ч
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146748
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Р
”
H__inference_sequential_8_layer_call_and_return_conditional_losses_146570
conv2d_16_input+
conv2d_16_146559:0А
conv2d_16_146561:	А+
conv2d_17_146564:А
conv2d_17_146566:
identityИҐ!conv2d_16/StatefulPartitionedCallҐ!conv2d_17/StatefulPartitionedCallЙ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputconv2d_16_146559conv2d_16_146561*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146449£
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_146564conv2d_17_146566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146465Б
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€О
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€0
)
_user_specified_nameconv2d_16_input
‘
г
-__inference_sequential_8_layer_call_fn_146787

inputs"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146532w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
Ж
»
__inference_call_113511
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Н
А
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146841

inputs9
conv2d_readvariableop_resource:0А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
Ѕ
№
+__inference_ca_model_8_layer_call_fn_146722
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146601w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Н
А
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146449

inputs9
conv2d_readvariableop_resource:0А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
‘
г
-__inference_sequential_8_layer_call_fn_146774

inputs"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146472w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
и
’
__inference_call_113498
x
n_times(
while_input_5:0А
while_input_6:	А(
while_input_7:А
while_input_8:
identityИҐwhileK
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
:€€€€€€€€€N
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : П
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
while_body_113366*
condR
while_cond_113365*7
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
«B
І

"__inference__traced_restore_146989
file_prefix<
!assignvariableop_conv2d_16_kernel:0А0
!assignvariableop_1_conv2d_16_bias:	А>
#assignvariableop_2_conv2d_17_kernel:А/
!assignvariableop_3_conv2d_17_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: E
*assignvariableop_8_adam_conv2d_16_kernel_m:0А7
(assignvariableop_9_adam_conv2d_16_bias_m:	АF
+assignvariableop_10_adam_conv2d_17_kernel_m:А7
)assignvariableop_11_adam_conv2d_17_bias_m:F
+assignvariableop_12_adam_conv2d_16_kernel_v:0А8
)assignvariableop_13_adam_conv2d_16_bias_v:	АF
+assignvariableop_14_adam_conv2d_17_kernel_v:А7
)assignvariableop_15_adam_conv2d_17_bias_v:
identity_17ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ю
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueЪBЧB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHТ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B у
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_17_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_17_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_8AssignVariableOp*assignvariableop_8_adam_conv2d_16_kernel_mIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_conv2d_16_bias_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_10AssignVariableOp+assignvariableop_10_adam_conv2d_17_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_conv2d_17_bias_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_12AssignVariableOp+assignvariableop_12_adam_conv2d_16_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_conv2d_16_bias_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_conv2d_17_kernel_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_conv2d_17_bias_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ѓ
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: Ь
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
Е
о
H__inference_sequential_8_layer_call_and_return_conditional_losses_146821

inputsC
(conv2d_16_conv2d_readvariableop_resource:0А8
)conv2d_16_biasadd_readvariableop_resource:	АC
(conv2d_17_conv2d_readvariableop_resource:А7
)conv2d_17_biasadd_readvariableop_resource:
identityИҐ conv2d_16/BiasAdd/ReadVariableOpҐconv2d_16/Conv2D/ReadVariableOpҐ conv2d_17/BiasAdd/ReadVariableOpҐconv2d_17/Conv2D/ReadVariableOpС
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0ѓ
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
З
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аm
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€АС
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ƒ
conv2d_17/Conv2DConv2Dconv2d_16/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
Ж
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ы
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€q
IdentityIdentityconv2d_17/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€–
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
П
®
__inference_forward_pass_113460
xP
5sequential_8_conv2d_16_conv2d_readvariableop_resource:0АE
6sequential_8_conv2d_16_biasadd_readvariableop_resource:	АP
5sequential_8_conv2d_17_conv2d_readvariableop_resource:АD
6sequential_8_conv2d_17_biasadd_readvariableop_resource:
identityИҐ-sequential_8/conv2d_16/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_16/Conv2D/ReadVariableOpҐ-sequential_8/conv2d_17/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_17/Conv2D/ReadVariableOpГ
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
GPU2*0J 8В *$
fR
__inference_perceive_113443Ђ
,sequential_8/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0“
sequential_8/conv2d_16/Conv2DConv2DPartitionedCall:output:04sequential_8/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:HHА*
paddingVALID*
strides
°
-sequential_8/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
sequential_8/conv2d_16/BiasAddBiasAdd&sequential_8/conv2d_16/Conv2D:output:05sequential_8/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:HHА~
sequential_8/conv2d_16/ReluRelu'sequential_8/conv2d_16/BiasAdd:output:0*
T0*'
_output_shapes
:HHАЂ
,sequential_8/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0в
sequential_8/conv2d_17/Conv2DConv2D)sequential_8/conv2d_16/Relu:activations:04sequential_8/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:HH*
paddingVALID*
strides
†
-sequential_8/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0є
sequential_8/conv2d_17/BiasAddBiasAdd&sequential_8/conv2d_17/Conv2D:output:05sequential_8/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:HHi
addAddV2x'sequential_8/conv2d_17/BiasAdd:output:0*
T0*&
_output_shapes
:HHU
IdentityIdentityadd:z:0^NoOp*
T0*&
_output_shapes
:HHД
NoOpNoOp.^sequential_8/conv2d_16/BiasAdd/ReadVariableOp-^sequential_8/conv2d_16/Conv2D/ReadVariableOp.^sequential_8/conv2d_17/BiasAdd/ReadVariableOp-^sequential_8/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:HH: : : : 2^
-sequential_8/conv2d_16/BiasAdd/ReadVariableOp-sequential_8/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_16/Conv2D/ReadVariableOp,sequential_8/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_17/BiasAdd/ReadVariableOp-sequential_8/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_17/Conv2D/ReadVariableOp,sequential_8/conv2d_17/Conv2D/ReadVariableOp:I E
&
_output_shapes
:HH

_user_specified_namex
µ
ч
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146640
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
«
э
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146690
input_1"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
у
°
*__inference_conv2d_16_layer_call_fn_146830

inputs"
unknown:0А
	unknown_0:	А
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146449x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
°	
Е
!__inference__wrapped_model_146431
input_1,
ca_model_8_146421:0А 
ca_model_8_146423:	А,
ca_model_8_146425:А
ca_model_8_146427:
identityИҐ"ca_model_8/StatefulPartitionedCall€
"ca_model_8/StatefulPartitionedCallStatefulPartitionedCallinput_1ca_model_8_146421ca_model_8_146423ca_model_8_146425ca_model_8_146427*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В * 
fR
__inference_call_113060В
IdentityIdentity+ca_model_8/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€k
NoOpNoOp#^ca_model_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2H
"ca_model_8/StatefulPartitionedCall"ca_model_8/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
І
џ
$__inference_signature_wrapper_146709
input_1"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_146431w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Г5
2
__inference_perceive_113443
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
value4B2"$   Њ       >  АЊ      А>   Њ       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$   Њ       >  АЊ      А>   Њ       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$                  А?                Е
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  А?                З
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis€€€€€€€€€l
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
valueB"            В
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
value	B :М
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
value	B :м
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:Д
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
valueB:т
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
valueB:ь
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
valueB:ш
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
value	B : ≈
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
valueB"      Щ
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
”
в
+__inference_ca_model_8_layer_call_fn_146612
input_1"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146601w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
І5
2
__inference_perceive_113745
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
value4B2"$   Њ       >  АЊ      А>   Њ       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$   Њ       >  АЊ      А>   Њ       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$                  А?                Е
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  А?                З
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis€€€€€€€€€l
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
valueB"            В
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
value	B :М
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
value	B :м
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:Д
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
valueB:т
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
valueB:ь
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
valueB:ш
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
value	B : ≈
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
valueB"      Ґ
	depthwiseDepthwiseConv2dNativexRepeat/Reshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€0*
paddingSAME*
strides
b
IdentityIdentitydepthwise:output:0*
T0*/
_output_shapes
:€€€€€€€€€0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Ж
»
__inference_call_113060
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
к
®
__inference_forward_pass_113568
xP
5sequential_8_conv2d_16_conv2d_readvariableop_resource:0АE
6sequential_8_conv2d_16_biasadd_readvariableop_resource:	АP
5sequential_8_conv2d_17_conv2d_readvariableop_resource:АD
6sequential_8_conv2d_17_biasadd_readvariableop_resource:
identityИҐ-sequential_8/conv2d_16/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_16/Conv2D/ReadVariableOpҐ-sequential_8/conv2d_17/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_17/Conv2D/ReadVariableOpМ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_perceive_113032Ђ
,sequential_8/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0џ
sequential_8/conv2d_16/Conv2DConv2DPartitionedCall:output:04sequential_8/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
°
-sequential_8/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0√
sequential_8/conv2d_16/BiasAddBiasAdd&sequential_8/conv2d_16/Conv2D:output:05sequential_8/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЗ
sequential_8/conv2d_16/ReluRelu'sequential_8/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
,sequential_8/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0л
sequential_8/conv2d_17/Conv2DConv2D)sequential_8/conv2d_16/Relu:activations:04sequential_8/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
†
-sequential_8/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_8/conv2d_17/BiasAddBiasAdd&sequential_8/conv2d_17/Conv2D:output:05sequential_8/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€r
addAddV2x'sequential_8/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€Д
NoOpNoOp.^sequential_8/conv2d_16/BiasAdd/ReadVariableOp-^sequential_8/conv2d_16/Conv2D/ReadVariableOp.^sequential_8/conv2d_17/BiasAdd/ReadVariableOp-^sequential_8/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2^
-sequential_8/conv2d_16/BiasAdd/ReadVariableOp-sequential_8/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_16/Conv2D/ReadVariableOp,sequential_8/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_17/BiasAdd/ReadVariableOp-sequential_8/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_17/Conv2D/ReadVariableOp,sequential_8/conv2d_17/Conv2D/ReadVariableOp:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
П
®
__inference_forward_pass_113549
xP
5sequential_8_conv2d_16_conv2d_readvariableop_resource:0АE
6sequential_8_conv2d_16_biasadd_readvariableop_resource:	АP
5sequential_8_conv2d_17_conv2d_readvariableop_resource:АD
6sequential_8_conv2d_17_biasadd_readvariableop_resource:
identityИҐ-sequential_8/conv2d_16/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_16/Conv2D/ReadVariableOpҐ-sequential_8/conv2d_17/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_17/Conv2D/ReadVariableOpГ
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
GPU2*0J 8В *$
fR
__inference_perceive_113443Ђ
,sequential_8/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0“
sequential_8/conv2d_16/Conv2DConv2DPartitionedCall:output:04sequential_8/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:HHА*
paddingVALID*
strides
°
-sequential_8/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
sequential_8/conv2d_16/BiasAddBiasAdd&sequential_8/conv2d_16/Conv2D:output:05sequential_8/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:HHА~
sequential_8/conv2d_16/ReluRelu'sequential_8/conv2d_16/BiasAdd:output:0*
T0*'
_output_shapes
:HHАЂ
,sequential_8/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0в
sequential_8/conv2d_17/Conv2DConv2D)sequential_8/conv2d_16/Relu:activations:04sequential_8/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:HH*
paddingVALID*
strides
†
-sequential_8/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0є
sequential_8/conv2d_17/BiasAddBiasAdd&sequential_8/conv2d_17/Conv2D:output:05sequential_8/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:HHi
addAddV2x'sequential_8/conv2d_17/BiasAdd:output:0*
T0*&
_output_shapes
:HHU
IdentityIdentityadd:z:0^NoOp*
T0*&
_output_shapes
:HHД
NoOpNoOp.^sequential_8/conv2d_16/BiasAdd/ReadVariableOp-^sequential_8/conv2d_16/Conv2D/ReadVariableOp.^sequential_8/conv2d_17/BiasAdd/ReadVariableOp-^sequential_8/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:HH: : : : 2^
-sequential_8/conv2d_16/BiasAdd/ReadVariableOp-sequential_8/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_16/Conv2D/ReadVariableOp,sequential_8/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_17/BiasAdd/ReadVariableOp-sequential_8/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_17/Conv2D/ReadVariableOp,sequential_8/conv2d_17/Conv2D/ReadVariableOp:I E
&
_output_shapes
:HH

_user_specified_namex
Р
”
H__inference_sequential_8_layer_call_and_return_conditional_losses_146584
conv2d_16_input+
conv2d_16_146573:0А
conv2d_16_146575:	А+
conv2d_17_146578:А
conv2d_17_146580:
identityИҐ!conv2d_16/StatefulPartitionedCallҐ!conv2d_17/StatefulPartitionedCallЙ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputconv2d_16_146573conv2d_16_146575*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146449£
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_146578conv2d_17_146580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146465Б
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€О
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€0
)
_user_specified_nameconv2d_16_input
≠

€
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146860

inputs9
conv2d_readvariableop_resource:А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
…
њ
while_cond_113365
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_less_maximum4
0while_while_cond_113365___redundant_placeholder04
0while_while_cond_113365___redundant_placeholder14
0while_while_cond_113365___redundant_placeholder24
0while_while_cond_113365___redundant_placeholder34
0while_while_cond_113365___redundant_placeholder4
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
”
в
+__inference_ca_model_8_layer_call_fn_146664
input_1"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146640w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
П
®
__inference_forward_pass_113530
xP
5sequential_8_conv2d_16_conv2d_readvariableop_resource:0АE
6sequential_8_conv2d_16_biasadd_readvariableop_resource:	АP
5sequential_8_conv2d_17_conv2d_readvariableop_resource:АD
6sequential_8_conv2d_17_biasadd_readvariableop_resource:
identityИҐ-sequential_8/conv2d_16/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_16/Conv2D/ReadVariableOpҐ-sequential_8/conv2d_17/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_17/Conv2D/ReadVariableOpГ
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
GPU2*0J 8В *$
fR
__inference_perceive_113032Ђ
,sequential_8/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0“
sequential_8/conv2d_16/Conv2DConv2DPartitionedCall:output:04sequential_8/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingVALID*
strides
°
-sequential_8/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
sequential_8/conv2d_16/BiasAddBiasAdd&sequential_8/conv2d_16/Conv2D:output:05sequential_8/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:А~
sequential_8/conv2d_16/ReluRelu'sequential_8/conv2d_16/BiasAdd:output:0*
T0*'
_output_shapes
:АЂ
,sequential_8/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0в
sequential_8/conv2d_17/Conv2DConv2D)sequential_8/conv2d_16/Relu:activations:04sequential_8/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
†
-sequential_8/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0є
sequential_8/conv2d_17/BiasAddBiasAdd&sequential_8/conv2d_17/Conv2D:output:05sequential_8/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:i
addAddV2x'sequential_8/conv2d_17/BiasAdd:output:0*
T0*&
_output_shapes
:U
IdentityIdentityadd:z:0^NoOp*
T0*&
_output_shapes
:Д
NoOpNoOp.^sequential_8/conv2d_16/BiasAdd/ReadVariableOp-^sequential_8/conv2d_16/Conv2D/ReadVariableOp.^sequential_8/conv2d_17/BiasAdd/ReadVariableOp-^sequential_8/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : 2^
-sequential_8/conv2d_16/BiasAdd/ReadVariableOp-sequential_8/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_16/Conv2D/ReadVariableOp,sequential_8/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_17/BiasAdd/ReadVariableOp-sequential_8/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_17/Conv2D/ReadVariableOp,sequential_8/conv2d_17/Conv2D/ReadVariableOp:I E
&
_output_shapes
:

_user_specified_namex
п
м
-__inference_sequential_8_layer_call_fn_146556
conv2d_16_input"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146532w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€0
)
_user_specified_nameconv2d_16_input
Г5
2
__inference_perceive_113627
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
value4B2"$   Њ       >  АЊ      А>   Њ       >L
mulMulCos:y:0mul/y:output:0*
T0*
_output_shapes

:|
mul_1/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$   Њ       >  АЊ      А>   Њ       >P
mul_2MulSin:y:0mul_2/y:output:0*
T0*
_output_shapes

:|
mul_3/yConst*
_output_shapes

:*
dtype0*=
value4B2"$   Њ  АЊ   Њ               >  А>   >P
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
value4B2"$                  А?                Е
stack_1/values_0Const*
_output_shapes

:*
dtype0*=
value4B2"$                  А?                З
stack_1Packstack_1/values_0:output:0sub:z:0add:z:0*
N*
T0*"
_output_shapes
:*
axis€€€€€€€€€l
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
valueB"            В
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
value	B :М
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
value	B :м
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0 Repeat/Tile/multiples/2:output:0Repeat/Reshape:output:0 Repeat/Tile/multiples/4:output:0*
N*
T0*
_output_shapes
:Д
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
valueB:т
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
valueB:ь
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
valueB:ш
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
value	B : ≈
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
valueB"      Щ
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
х
 
H__inference_sequential_8_layer_call_and_return_conditional_losses_146472

inputs+
conv2d_16_146450:0А
conv2d_16_146452:	А+
conv2d_17_146466:А
conv2d_17_146468:
identityИҐ!conv2d_16/StatefulPartitionedCallҐ!conv2d_17/StatefulPartitionedCallА
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_146450conv2d_16_146452*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146449£
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_146466conv2d_17_146468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146465Б
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€О
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€0: : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
µ
ч
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146601
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
в
»
__inference_call_113347
x"
unknown:0А
	unknown_0:	А$
	unknown_1:А
	unknown_2:
identityИҐStatefulPartitionedCallЋ
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
GPU2*0J 8В *(
f#R!
__inference_forward_pass_113049n
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
к
®
__inference_forward_pass_113049
xP
5sequential_8_conv2d_16_conv2d_readvariableop_resource:0АE
6sequential_8_conv2d_16_biasadd_readvariableop_resource:	АP
5sequential_8_conv2d_17_conv2d_readvariableop_resource:АD
6sequential_8_conv2d_17_biasadd_readvariableop_resource:
identityИҐ-sequential_8/conv2d_16/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_16/Conv2D/ReadVariableOpҐ-sequential_8/conv2d_17/BiasAdd/ReadVariableOpҐ,sequential_8/conv2d_17/Conv2D/ReadVariableOpМ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_perceive_113032Ђ
,sequential_8/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:0А*
dtype0џ
sequential_8/conv2d_16/Conv2DConv2DPartitionedCall:output:04sequential_8/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
°
-sequential_8/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0√
sequential_8/conv2d_16/BiasAddBiasAdd&sequential_8/conv2d_16/Conv2D:output:05sequential_8/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЗ
sequential_8/conv2d_16/ReluRelu'sequential_8/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
,sequential_8/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0л
sequential_8/conv2d_17/Conv2DConv2D)sequential_8/conv2d_16/Relu:activations:04sequential_8/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
†
-sequential_8/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_8/conv2d_17/BiasAddBiasAdd&sequential_8/conv2d_17/Conv2D:output:05sequential_8/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€r
addAddV2x'sequential_8/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€Д
NoOpNoOp.^sequential_8/conv2d_16/BiasAdd/ReadVariableOp-^sequential_8/conv2d_16/Conv2D/ReadVariableOp.^sequential_8/conv2d_17/BiasAdd/ReadVariableOp-^sequential_8/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : 2^
-sequential_8/conv2d_16/BiasAdd/ReadVariableOp-sequential_8/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_16/Conv2D/ReadVariableOp,sequential_8/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_17/BiasAdd/ReadVariableOp-sequential_8/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_17/Conv2D/ReadVariableOp,sequential_8/conv2d_17/Conv2D/ReadVariableOp:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
√*
С
__inference__traced_save_146931
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ы
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueЪBЧB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHП
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B •
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*Ї
_input_shapes®
•: :0А:А:А:: : : : :0А:А:А::0А:А:А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:0А:!

_output_shapes	
:А:-)
'
_output_shapes
:А: 
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
:0А:!


_output_shapes	
:А:-)
'
_output_shapes
:А: 

_output_shapes
::-)
'
_output_shapes
:0А:!

_output_shapes	
:А:-)
'
_output_shapes
:А: 

_output_shapes
::

_output_shapes
: "µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ї
serving_defaultІ
C
input_18
serving_default_input_1:0€€€€€€€€€D
output_18
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Ѕ©
Ч
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
 
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
м
trace_0
trace_1
trace_2
trace_32Б
+__inference_ca_model_8_layer_call_fn_146612
+__inference_ca_model_8_layer_call_fn_146722
+__inference_ca_model_8_layer_call_fn_146735
+__inference_ca_model_8_layer_call_fn_146664 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ў
trace_0
trace_1
trace_2
trace_32н
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146748
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146761
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146677
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146690 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ћB…
!__inference__wrapped_model_146431input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ш
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
И
'iter

(beta_1

)beta_2
	*decaym^m_m`mavbvcvdve"
	optimizer
ћ
+trace_0
,trace_1
-trace_22ы
__inference_call_113347
__inference_call_113498
__inference_call_113511≠
§≤†
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z+trace_0z,trace_1z-trace_2
й
.trace_0
/trace_1
0trace_22Ш
__inference_forward_pass_113530
__inference_forward_pass_113549
__inference_forward_pass_113568≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z.trace_0z/trace_1z0trace_2
Ё
1trace_0
2trace_1
3trace_22М
__inference_perceive_113627
__inference_perceive_113686
__inference_perceive_113745≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z1trace_0z2trace_1z3trace_2
,
4serving_default"
signature_map
+:)0А2conv2d_16/kernel
:А2conv2d_16/bias
+:)А2conv2d_17/kernel
:2conv2d_17/bias
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
ИBЕ
+__inference_ca_model_8_layer_call_fn_146612input_1" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ВB€
+__inference_ca_model_8_layer_call_fn_146722x" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ВB€
+__inference_ca_model_8_layer_call_fn_146735x" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ИBЕ
+__inference_ca_model_8_layer_call_fn_146664input_1" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЭBЪ
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146748x" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЭBЪ
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146761x" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
£B†
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146677input_1" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
£B†
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146690input_1" 
Ѕ≤љ
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
Ё
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
Ё
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
≠
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
й
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32ю
-__inference_sequential_8_layer_call_fn_146483
-__inference_sequential_8_layer_call_fn_146774
-__inference_sequential_8_layer_call_fn_146787
-__inference_sequential_8_layer_call_fn_146556њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
’
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32к
H__inference_sequential_8_layer_call_and_return_conditional_losses_146804
H__inference_sequential_8_layer_call_and_return_conditional_losses_146821
H__inference_sequential_8_layer_call_and_return_conditional_losses_146570
H__inference_sequential_8_layer_call_and_return_conditional_losses_146584њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
—Bќ
__inference_call_113347x"≠
§≤†
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЏB„
__inference_call_113498xn_times"≠
§≤†
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—Bќ
__inference_call_113511x"≠
§≤†
FullArgSpec#
argsЪ
jself
jx
	jn_times
varargs
 
varkw
 
defaultsҐ
`

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ёBџ
__inference_forward_pass_113530x"≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ёBџ
__inference_forward_pass_113549x"≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ёBџ
__inference_forward_pass_113568x"≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЏB„
__inference_perceive_113627x"≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЏB„
__inference_perceive_113686x"≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЏB„
__inference_perceive_113745x"≤
©≤•
FullArgSpec!
argsЪ
jself
jx
jangle
varargs
 
varkw
 
defaultsҐ
	Y        

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЋB»
$__inference_signature_wrapper_146709input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Utrace_02—
*__inference_conv2d_16_layer_call_fn_146830Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zUtrace_0
Й
Vtrace_02м
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146841Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zVtrace_0
і2±Ѓ
£≤Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
\trace_02—
*__inference_conv2d_17_layer_call_fn_146850Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z\trace_0
Й
]trace_02м
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146860Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z]trace_0
і2±Ѓ
£≤Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЗBД
-__inference_sequential_8_layer_call_fn_146483conv2d_16_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
-__inference_sequential_8_layer_call_fn_146774inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
-__inference_sequential_8_layer_call_fn_146787inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
-__inference_sequential_8_layer_call_fn_146556conv2d_16_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146804inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146821inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ҐBЯ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146570conv2d_16_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ҐBЯ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146584conv2d_16_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_conv2d_16_layer_call_fn_146830inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146841inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_conv2d_17_layer_call_fn_146850inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146860inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0:.0А2Adam/conv2d_16/kernel/m
": А2Adam/conv2d_16/bias/m
0:.А2Adam/conv2d_17/kernel/m
!:2Adam/conv2d_17/bias/m
0:.0А2Adam/conv2d_16/kernel/v
": А2Adam/conv2d_16/bias/v
0:.А2Adam/conv2d_17/kernel/v
!:2Adam/conv2d_17/bias/vҐ
!__inference__wrapped_model_146431}8Ґ5
.Ґ+
)К&
input_1€€€€€€€€€
™ ";™8
6
output_1*К'
output_1€€€€€€€€€ќ
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146677ГLҐI
2Ґ/
)К&
input_1€€€€€€€€€
`
™

trainingp "-Ґ*
#К 
0€€€€€€€€€
Ъ ќ
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146690ГLҐI
2Ґ/
)К&
input_1€€€€€€€€€
`
™

trainingp"-Ґ*
#К 
0€€€€€€€€€
Ъ «
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146748}FҐC
,Ґ)
#К 
x€€€€€€€€€
`
™

trainingp "-Ґ*
#К 
0€€€€€€€€€
Ъ «
F__inference_ca_model_8_layer_call_and_return_conditional_losses_146761}FҐC
,Ґ)
#К 
x€€€€€€€€€
`
™

trainingp"-Ґ*
#К 
0€€€€€€€€€
Ъ •
+__inference_ca_model_8_layer_call_fn_146612vLҐI
2Ґ/
)К&
input_1€€€€€€€€€
`
™

trainingp " К€€€€€€€€€•
+__inference_ca_model_8_layer_call_fn_146664vLҐI
2Ґ/
)К&
input_1€€€€€€€€€
`
™

trainingp" К€€€€€€€€€Я
+__inference_ca_model_8_layer_call_fn_146722pFҐC
,Ґ)
#К 
x€€€€€€€€€
`
™

trainingp " К€€€€€€€€€Я
+__inference_ca_model_8_layer_call_fn_146735pFҐC
,Ґ)
#К 
x€€€€€€€€€
`
™

trainingp" К€€€€€€€€€i
__inference_call_113347N-Ґ*
#Ґ 
К
x
`
™ "Кw
__inference_call_113498\;Ґ8
1Ґ.
К
xHH
К
n_times 
™ "КHH{
__inference_call_113511`6Ґ3
,Ґ)
#К 
x€€€€€€€€€
`
™ " К€€€€€€€€€ґ
E__inference_conv2d_16_layer_call_and_return_conditional_losses_146841m7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€0
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ О
*__inference_conv2d_16_layer_call_fn_146830`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€0
™ "!К€€€€€€€€€Аґ
E__inference_conv2d_17_layer_call_and_return_conditional_losses_146860m8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ О
*__inference_conv2d_17_layer_call_fn_146850`8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ " К€€€€€€€€€x
__inference_forward_pass_113530U4Ґ1
*Ґ'
К
x
	Y        
™ "Кx
__inference_forward_pass_113549U4Ґ1
*Ґ'
К
xHH
	Y        
™ "КHHК
__inference_forward_pass_113568g=Ґ:
3Ґ0
#К 
x€€€€€€€€€
	Y        
™ " К€€€€€€€€€n
__inference_perceive_113627O4Ґ1
*Ґ'
К
x
	Y        
™ "К0n
__inference_perceive_113686O4Ґ1
*Ґ'
К
xHH
	Y        
™ "КHH0А
__inference_perceive_113745a=Ґ:
3Ґ0
#К 
x€€€€€€€€€
	Y        
™ " К€€€€€€€€€0Ћ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146570HҐE
>Ґ;
1К.
conv2d_16_input€€€€€€€€€0
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Ћ
H__inference_sequential_8_layer_call_and_return_conditional_losses_146584HҐE
>Ґ;
1К.
conv2d_16_input€€€€€€€€€0
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ¬
H__inference_sequential_8_layer_call_and_return_conditional_losses_146804v?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€0
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ¬
H__inference_sequential_8_layer_call_and_return_conditional_losses_146821v?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€0
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ £
-__inference_sequential_8_layer_call_fn_146483rHҐE
>Ґ;
1К.
conv2d_16_input€€€€€€€€€0
p 

 
™ " К€€€€€€€€€£
-__inference_sequential_8_layer_call_fn_146556rHҐE
>Ґ;
1К.
conv2d_16_input€€€€€€€€€0
p

 
™ " К€€€€€€€€€Ъ
-__inference_sequential_8_layer_call_fn_146774i?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€0
p 

 
™ " К€€€€€€€€€Ъ
-__inference_sequential_8_layer_call_fn_146787i?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€0
p

 
™ " К€€€€€€€€€±
$__inference_signature_wrapper_146709ИCҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€";™8
6
output_1*К'
output_1€€€€€€€€€