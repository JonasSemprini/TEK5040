ц─
Р▒
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceѕ
«
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
е
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8Ко
њ
ConstConst*
_output_shapes

:*
dtype0*U
valueLBJ"<  ђ┐                      ђ?              ђ?              ђ?
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђп*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ђп*
dtype0
і
serving_default_input_1Placeholder*/
_output_shapes
:         ``*
dtype0*$
shape:         ``
О
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_35358

NoOpNoOp
░
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*ж
value▀B▄ BН
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
policy_network
	
signatures*


0
1*


0
1*
* 
░
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 

	capture_2* 
▓
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
feature_extractor
	dense*

serving_default* 
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
* 
* 
* 

	capture_2* 

	capture_2* 
* 


0
1*


0
1*
* 
Њ
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

"trace_0* 

#trace_0* 
Џ
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*flatten* 
д
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses


kernel
bias*

	capture_2* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
Љ
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

6trace_0* 

7trace_0* 
ј
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 


0
1*


0
1*
* 
Њ
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
* 
	
*0* 
* 
* 
* 
* 
* 
* 
* 
* 
Љ
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Jtrace_0* 

Ktrace_0* 
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
И
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasConst_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *'
f"R 
__inference__traced_save_35423
▒
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_35438Ђ░
┬
┼
I__inference_policy_network_layer_call_and_return_conditional_losses_35305
input_1
dense_35296:
ђп
dense_35298:
identity	ѕбdense/StatefulPartitionedCall╠
!feature_extractor/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђп* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_feature_extractor_layer_call_and_return_conditional_losses_35272Ё
dense/StatefulPartitionedCallStatefulPartitionedCall*feature_extractor/PartitionedCall:output:0dense_35296dense_35298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_35295e
#categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :«
categorical/MultinomialMultinomial&dense/StatefulPartitionedCall:output:0,categorical/Multinomial/num_samples:output:0*
T0*'
_output_shapes
:         ѓ
SqueezeSqueeze categorical/Multinomial:output:0*
T0	*#
_output_shapes
:         *
squeeze_dims

         [
IdentityIdentitySqueeze:output:0^NoOp*
T0	*#
_output_shapes
:         B
NoOpNoOp^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ``: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:%!

_user_specified_name35298:%!

_user_specified_name35296:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
к
^
B__inference_flatten_layer_call_and_return_conditional_losses_35269

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     l  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђпZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђп"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ``:W S
/
_output_shapes
:         ``
 
_user_specified_nameinputs
ч	
з
@__inference_dense_layer_call_and_return_conditional_losses_35377

inputs2
matmul_readvariableop_resource:
ђп-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђп*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђп: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:         ђп
 
_user_specified_nameinputs
─
N
1__inference_feature_extractor_layer_call_fn_35277
input_1
identity║
PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђп* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_feature_extractor_layer_call_and_return_conditional_losses_35272b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђп"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ``:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
и
╗
__inference__traced_save_35423
file_prefix7
#read_disablecopyonread_dense_kernel:
ђп1
#read_1_disablecopyonread_dense_bias:
savev2_const_1

identity_5ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 А
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђп*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђпc

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђпw
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 Ъ
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:п
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ђ
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B є
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0savev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_4Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_5IdentityIdentity_4:output:0^NoOp*
T0*
_output_shapes
: Ў
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_1:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
─
ц
%__inference_agent_layer_call_fn_35346
input_1
unknown:
ђп
	unknown_0:
	unknown_1
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_agent_layer_call_and_return_conditional_losses_35335o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ``: : :22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::%!

_user_specified_name35340:%!

_user_specified_name35338:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
Ѓ
ъ
.__inference_policy_network_layer_call_fn_35314
input_1
unknown:
ђп
	unknown_0:
identity	ѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_policy_network_layer_call_and_return_conditional_losses_35305k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ``: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name35310:%!

_user_specified_name35308:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
┌
¤
!__inference__traced_restore_35438
file_prefix1
assignvariableop_dense_kernel:
ђп+
assignvariableop_1_dense_bias:

identity_3ѕбAssignVariableOpбAssignVariableOp_1█
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ђ
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B Г
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ѓ

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: L
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2(
AssignVariableOp_1AssignVariableOp_12$
AssignVariableOpAssignVariableOp:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ы
█
 __inference__wrapped_model_35261
input_1M
9agent_policy_network_dense_matmul_readvariableop_resource:
ђпH
:agent_policy_network_dense_biasadd_readvariableop_resource:
agent_gatherv2_params
identityѕб1agent/policy_network/dense/BiasAdd/ReadVariableOpб0agent/policy_network/dense/MatMul/ReadVariableOpЁ
4agent/policy_network/feature_extractor/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     l  й
6agent/policy_network/feature_extractor/flatten/ReshapeReshapeinput_1=agent/policy_network/feature_extractor/flatten/Const:output:0*
T0*)
_output_shapes
:         ђпг
0agent/policy_network/dense/MatMul/ReadVariableOpReadVariableOp9agent_policy_network_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђп*
dtype0п
!agent/policy_network/dense/MatMulMatMul?agent/policy_network/feature_extractor/flatten/Reshape:output:08agent/policy_network/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         е
1agent/policy_network/dense/BiasAdd/ReadVariableOpReadVariableOp:agent_policy_network_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0К
"agent/policy_network/dense/BiasAddBiasAdd+agent/policy_network/dense/MatMul:product:09agent/policy_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
8agent/policy_network/categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :П
,agent/policy_network/categorical/MultinomialMultinomial+agent/policy_network/dense/BiasAdd:output:0Aagent/policy_network/categorical/Multinomial/num_samples:output:0*
T0*'
_output_shapes
:         г
agent/policy_network/SqueezeSqueeze5agent/policy_network/categorical/Multinomial:output:0*
T0	*#
_output_shapes
:         *
squeeze_dims

         U
agent/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
agent/GatherV2GatherV2agent_gatherv2_params%agent/policy_network/Squeeze:output:0agent/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         f
IdentityIdentityagent/GatherV2:output:0^NoOp*
T0*'
_output_shapes
:         Ѕ
NoOpNoOp2^agent/policy_network/dense/BiasAdd/ReadVariableOp1^agent/policy_network/dense/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ``: : :2f
1agent/policy_network/dense/BiasAdd/ReadVariableOp1agent/policy_network/dense/BiasAdd/ReadVariableOp2d
0agent/policy_network/dense/MatMul/ReadVariableOp0agent/policy_network/dense/MatMul/ReadVariableOp:$ 

_output_shapes

::($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
к
^
B__inference_flatten_layer_call_and_return_conditional_losses_35388

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     l  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђпZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђп"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ``:W S
/
_output_shapes
:         ``
 
_user_specified_nameinputs
р
В
@__inference_agent_layer_call_and_return_conditional_losses_35335
input_1(
policy_network_35326:
ђп"
policy_network_35328:
gatherv2_params
identityѕб&policy_network/StatefulPartitionedCallѓ
&policy_network/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_network_35326policy_network_35328*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_policy_network_layer_call_and_return_conditional_losses_35305O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ├
GatherV2GatherV2gatherv2_params/policy_network/StatefulPartitionedCall:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         `
IdentityIdentityGatherV2:output:0^NoOp*
T0*'
_output_shapes
:         K
NoOpNoOp'^policy_network/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ``: : :2P
&policy_network/StatefulPartitionedCall&policy_network/StatefulPartitionedCall:$ 

_output_shapes

::%!

_user_specified_name35328:%!

_user_specified_name35326:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
ч	
з
@__inference_dense_layer_call_and_return_conditional_losses_35295

inputs2
matmul_readvariableop_resource:
ђп-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђп*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђп: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:         ђп
 
_user_specified_nameinputs
Ж
ћ
%__inference_dense_layer_call_fn_35367

inputs
unknown:
ђп
	unknown_0:
identityѕбStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_35295o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђп: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name35363:%!

_user_specified_name35361:Q M
)
_output_shapes
:         ђп
 
_user_specified_nameinputs
Г
C
'__inference_flatten_layer_call_fn_35382

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђп* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_35269b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђп"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ``:W S
/
_output_shapes
:         ``
 
_user_specified_nameinputs
б
б
#__inference_signature_wrapper_35358
input_1
unknown:
ђп
	unknown_0:
	unknown_1
identityѕбStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_35261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ``: : :22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::%!

_user_specified_name35352:%!

_user_specified_name35350:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1
т
i
L__inference_feature_extractor_layer_call_and_return_conditional_losses_35272
input_1
identityИ
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђп* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_35269j
IdentityIdentity flatten/PartitionedCall:output:0*
T0*)
_output_shapes
:         ђп"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ``:X T
/
_output_shapes
:         ``
!
_user_specified_name	input_1"ДL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
C
input_18
serving_default_input_1:0         ``<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:їY
Т
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
policy_network
	
signatures"
_tf_keras_model
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
С
trace_02К
%__inference_agent_layer_call_fn_35346Ю
ќ▓њ
FullArgSpec
argsџ
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
 
trace_02Р
@__inference_agent_layer_call_and_return_conditional_losses_35335Ю
ќ▓њ
FullArgSpec
argsџ
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0
ж
	capture_2B╚
 __inference__wrapped_model_35261input_1"ў
Љ▓Ї
FullArgSpec
argsџ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z	capture_2
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
feature_extractor
	dense"
_tf_keras_model
,
serving_default"
signature_map
 :
ђп2dense/kernel
:2
dense/bias
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
з
	capture_2Bм
%__inference_agent_layer_call_fn_35346input_1"Ю
ќ▓њ
FullArgSpec
argsџ
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z	capture_2
ј
	capture_2Bь
@__inference_agent_layer_call_and_return_conditional_losses_35335input_1"Ю
ќ▓њ
FullArgSpec
argsџ
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z	capture_2
J
Constjtf.TrackableConstant
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
с
"trace_02к
.__inference_policy_network_layer_call_fn_35314Њ
ї▓ѕ
FullArgSpec
argsџ
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z"trace_0
■
#trace_02р
I__inference_policy_network_layer_call_and_return_conditional_losses_35305Њ
ї▓ѕ
FullArgSpec
argsџ
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z#trace_0
▓
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*flatten"
_tf_keras_model
╗
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses


kernel
bias"
_tf_keras_layer
ь
	capture_2B╠
#__inference_signature_wrapper_35358input_1"Ў
њ▓ј
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ
	jinput_1
kwonlydefaults
 
annotationsф *
 z	capture_2
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBЛ
.__inference_policy_network_layer_call_fn_35314input_1"Њ
ї▓ѕ
FullArgSpec
argsџ
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
I__inference_policy_network_layer_call_and_return_conditional_losses_35305input_1"Њ
ї▓ѕ
FullArgSpec
argsџ
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ч
6trace_02▀
1__inference_feature_extractor_layer_call_fn_35277Е
б▓ъ
FullArgSpec!
argsџ
jx
jsample_action
varargs
 
varkw
 
defaultsб
p

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z6trace_0
Ќ
7trace_02Щ
L__inference_feature_extractor_layer_call_and_return_conditional_losses_35272Е
б▓ъ
FullArgSpec!
argsџ
jx
jsample_action
varargs
 
varkw
 
defaultsб
p

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z7trace_0
Ц
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
▀
Ctrace_02┬
%__inference_dense_layer_call_fn_35367ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zCtrace_0
Щ
Dtrace_02П
@__inference_dense_layer_call_and_return_conditional_losses_35377ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zDtrace_0
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
УBт
1__inference_feature_extractor_layer_call_fn_35277input_1"ц
Ю▓Ў
FullArgSpec!
argsџ
jx
jsample_action
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
L__inference_feature_extractor_layer_call_and_return_conditional_losses_35272input_1"ц
Ю▓Ў
FullArgSpec!
argsџ
jx
jsample_action
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
р
Jtrace_02─
'__inference_flatten_layer_call_fn_35382ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zJtrace_0
Ч
Ktrace_02▀
B__inference_flatten_layer_call_and_return_conditional_losses_35388ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zKtrace_0
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
¤B╠
%__inference_dense_layer_call_fn_35367inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЖBу
@__inference_dense_layer_call_and_return_conditional_losses_35377inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ЛB╬
'__inference_flatten_layer_call_fn_35382inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ВBж
B__inference_flatten_layer_call_and_return_conditional_losses_35388inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ў
 __inference__wrapped_model_35261t
8б5
.б+
)і&
input_1         ``
ф "3ф0
.
output_1"і
output_1         ▒
@__inference_agent_layer_call_and_return_conditional_losses_35335m
8б5
.б+
)і&
input_1         ``
ф ",б)
"і
tensor_0         
џ І
%__inference_agent_layer_call_fn_35346b
8б5
.б+
)і&
input_1         ``
ф "!і
unknown         Е
@__inference_dense_layer_call_and_return_conditional_losses_35377e
1б.
'б$
"і
inputs         ђп
ф ",б)
"і
tensor_0         
џ Ѓ
%__inference_dense_layer_call_fn_35367Z
1б.
'б$
"і
inputs         ђп
ф "!і
unknown         Й
L__inference_feature_extractor_layer_call_and_return_conditional_losses_35272n<б9
2б/
)і&
input_1         ``
p
ф ".б+
$і!
tensor_0         ђп
џ ў
1__inference_feature_extractor_layer_call_fn_35277c<б9
2б/
)і&
input_1         ``
p
ф "#і 
unknown         ђп»
B__inference_flatten_layer_call_and_return_conditional_losses_35388i7б4
-б*
(і%
inputs         ``
ф ".б+
$і!
tensor_0         ђп
џ Ѕ
'__inference_flatten_layer_call_fn_35382^7б4
-б*
(і%
inputs         ``
ф "#і 
unknown         ђпх
I__inference_policy_network_layer_call_and_return_conditional_losses_35305h
8б5
.б+
)і&
input_1         ``
ф "(б%
і
tensor_0         	
џ Ј
.__inference_policy_network_layer_call_fn_35314]
8б5
.б+
)і&
input_1         ``
ф "і
unknown         	д
#__inference_signature_wrapper_35358
Cб@
б 
9ф6
4
input_1)і&
input_1         ``"3ф0
.
output_1"і
output_1         